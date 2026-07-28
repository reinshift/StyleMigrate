/*
  StyleMigrate - 前端任意风格迁移（TensorFlow.js + @magenta/image）
  说明：完全在浏览器本地运行，不上传图片到服务器。
  优化：内存管理、低端设备兼容、推理超时、资源清理
*/

(() => {
  const els = {
    contentInput: document.getElementById('contentInput'),
    styleInput: document.getElementById('styleInput'),
    contentPreview: document.getElementById('contentPreview'),
    stylePreview: document.getElementById('stylePreview'),
    runBtn: document.getElementById('runBtn'),
    fallbackBtn: document.getElementById('fallbackBtn'),
    downloadBtn: document.getElementById('downloadBtn'),
    status: document.getElementById('status'),
    resultCanvas: document.getElementById('resultCanvas'),
  };

  const MAX_SIDE = 1024;
  let model = null;
  let modelReady = false;
  let modelLoading = false;
  let resultReady = false;
  let isRunning = false; // 防止并发推理

  // ============ 工具函数 ============

  // 带超时的 Promise 包装
  function withTimeout(promise, ms, label) {
    let timer;
    const timeout = new Promise((_, reject) => {
      timer = setTimeout(() => reject(
        new Error((label || '操作') + '超时（' + Math.round(ms / 1000) + '秒），设备性能可能不足')
      ), ms);
    });
    return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
  }

  // 简单重试工具（指数退避）
  async function withRetry(taskFn, { retries = 2, baseDelay = 400, onAttempt, timeoutMs } = {}) {
    let lastErr;
    for (let attempt = 0; attempt <= retries; attempt++) {
      try {
        if (onAttempt) onAttempt(attempt);
        const p = taskFn();
        return timeoutMs ? await withTimeout(p, timeoutMs, '推理') : await p;
      } catch (e) {
        lastErr = e;
        // 超时或 OOM 类错误不重试
        if (e.message && (e.message.includes('超时') || e.message.includes('out of memory') || e.message.includes('OOM'))) break;
        if (attempt === retries) break;
        const delay = baseDelay * Math.pow(2, attempt);
        await new Promise(r => setTimeout(r, delay));
      }
    }
    throw lastErr;
  }

  function getDeviceMemoryGB() {
    try { return Math.max(0, Number(navigator.deviceMemory || 0)); } catch (_) { return 0; }
  }

  function getHardwareConcurrency() {
    try { return navigator.hardwareConcurrency || 2; } catch (_) { return 2; }
  }

  // 判断是否为低端设备（与诊断面板逻辑一致）
  function isLowEndDevice() {
    const mem = getDeviceMemoryGB();
    const cores = getHardwareConcurrency();
    // 内存 ≤3GB 或 核心数 ≤2 视为低端
    return (mem > 0 && mem <= 3) || cores <= 2;
  }

  function pickSizesByBackend(backend) {
    const mem = getDeviceMemoryGB();
    const lowEnd = isLowEndDevice();
    const veryLow = mem > 0 && mem <= 2; // 极低端
    if (backend === 'webgl' || backend === 'webgpu') {
      if (veryLow) return { content: 320, style: 192 };
      if (lowEnd) return { content: 512, style: 256 };
      if (mem && mem <= 4) return { content: 768, style: 384 };
      return { content: 1024, style: 512 };
    }
    if (backend === 'wasm') {
      if (veryLow) return { content: 256, style: 160 };
      if (lowEnd) return { content: 384, style: 192 };
      if (mem && mem <= 4) return { content: 512, style: 256 };
      return { content: 640, style: 384 };
    }
    // cpu 兜底
    if (veryLow) return { content: 256, style: 160 };
    if (lowEnd) return { content: 320, style: 192 };
    return { content: 448, style: 288 };
  }

  // 推理前内存检查（仅 Chrome 有效）
  function checkMemoryBeforeRun() {
    if (performance && performance.memory) {
      const used = performance.memory.usedJSHeapSize;
      const limit = performance.memory.jsHeapSizeLimit;
      const availRatio = (limit - used) / limit;
      if (availRatio < 0.15) {
        console.warn('可用堆内存不足 ' + Math.round(availRatio * 100) + '%，可能 OOM');
        return false;
      }
    }
    return true;
  }

  // 清理推理产生的临时资源
  function cleanupInferenceResources(refs) {
    try {
      if (refs) {
        if (refs.contentCanvas) { refs.contentCanvas.width = 0; refs.contentCanvas.height = 0; }
        if (refs.styleCanvas) { refs.styleCanvas.width = 0; refs.styleCanvas.height = 0; }
        refs.contentCanvas = null;
        refs.styleCanvas = null;
        refs.contentImg = null;
        refs.styleImg = null;
      }
    } catch (_) {}
  }

  function setStatus(text) {
    els.status.textContent = text;
  }

  function enableRunIfReady() {
    const hasContent = !!els.contentPreview.src;
    const hasStyle = !!els.stylePreview.src;
    const ready = hasContent && hasStyle;
    els.runBtn.disabled = !ready;
    els.fallbackBtn.disabled = !ready;
    setStatus(ready ? '点击"开始风格迁移"（或"轻量模式"）。' : '请先选择内容图与风格图。');
  }

  function fileToImage(file) {
    return new Promise((resolve, reject) => {
      const fr = new FileReader();
      fr.onerror = () => reject(new Error('读取文件失败'));
      fr.onload = () => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error('图片加载失败'));
        img.src = fr.result;
      };
      fr.readAsDataURL(file);
    });
  }

  function downscaleToCanvas(img, maxSide = MAX_SIDE) {
    const { width, height } = img;
    let w = width, h = height;
    if (Math.max(width, height) > maxSide) {
      if (width >= height) {
        w = maxSide;
        h = Math.round((height / width) * maxSide);
      } else {
        h = maxSide;
        w = Math.round((width / height) * maxSide);
      }
    }
    const canvas = document.createElement('canvas');
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, w, h);
    return canvas;
  }

  // ============ 模型加载 ============

  async function ensureModel() {
    if (model) return model;
    if (!(window.tf && window.tf.engine)) {
      throw new Error('TensorFlow.js 未正确加载，请刷新页面重试');
    }
    if (!(window.mi && window.mi.ArbitraryStyleTransferNetwork)) {
      throw new Error('@magenta/image 未正确加载，请刷新页面重试');
    }

    // 后端优先级：webgl/webgpu -> wasm -> cpu
    let backend = tf.getBackend();
    try {
      if (backend !== 'webgl' && backend !== 'webgpu') {
        await tf.setBackend('webgl');
        await tf.ready();
        backend = tf.getBackend();
      }
    } catch (_) {}
    if (backend !== 'webgl' && backend !== 'webgpu') {
      try {
        if (tf.wasm && typeof tf.wasm.setWasmPaths === 'function') {
          tf.wasm.setWasmPaths('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@3.21.0/dist/');
        }
        await tf.setBackend('wasm');
        await tf.ready();
        backend = tf.getBackend();
      } catch (_) {}
    }
    if (backend !== 'webgl' && backend !== 'webgpu' && backend !== 'wasm') {
      try { await tf.setBackend('cpu'); await tf.ready(); backend = tf.getBackend(); }
      catch (_) {}
    }

    // 监听 WebGL 上下文丢失
    if (backend === 'webgl') {
      const glCanvas = document.querySelector('.canvas-wrap canvas');
      if (glCanvas) {
        glCanvas.addEventListener('webglcontextlost', function onCtxLost(e) {
          e.preventDefault();
          console.warn('WebGL 上下文已丢失');
          setStatus('⚠️ WebGL 上下文丢失，正在尝试恢复…');
          model = null; modelReady = false;
          setTimeout(async () => {
            try {
              await tf.setBackend('webgl');
              await tf.ready();
              await ensureModel();
              setStatus('✅ WebGL 已恢复，模型就绪');
            } catch (_) {
              setStatus('⚠️ WebGL 恢复失败，请刷新页面');
            }
          }, 1000);
        }, { once: true });
      }
    }

    setStatus('正在加载模型…（后端：' + backend + '）');
    modelLoading = true;
    model = new mi.ArbitraryStyleTransferNetwork();
    if (typeof model.initialize === 'function') {
      await withRetry(() => model.initialize(), {
        retries: 1,
        baseDelay: 800,
        timeoutMs: 60000,
        onAttempt: (k) => setStatus('正在加载模型…（后端：' + backend + (k ? '，重试 ' + (k + 1) : '') + '）')
      });
    }
    modelReady = true;
    modelLoading = false;
    setStatus('模型就绪（后端：' + backend + '）');
    return model;
  }

  // ============ 轻量降级方案：纯 Canvas 色彩风格迁移 ============

  // RGB ↔ LAB 转换（用于色彩统计匹配）
  function rgbToLab(r, g, b) {
    // sRGB → linear
    let rl = r / 255, gl = g / 255, bl = b / 255;
    rl = rl > 0.04045 ? Math.pow((rl + 0.055) / 1.055, 2.4) : rl / 12.92;
    gl = gl > 0.04045 ? Math.pow((gl + 0.055) / 1.055, 2.4) : gl / 12.92;
    bl = bl > 0.04045 ? Math.pow((bl + 0.055) / 1.055, 2.4) : bl / 12.92;
    // linear → XYZ (D65)
    let x = (rl * 0.4124564 + gl * 0.3575761 + bl * 0.1804375) / 0.95047;
    let y = (rl * 0.2126729 + gl * 0.7151522 + bl * 0.0721750) / 1.00000;
    let z = (rl * 0.0193339 + gl * 0.1191920 + bl * 0.9503041) / 1.08883;
    // XYZ → LAB
    const f = (t) => t > 0.008856 ? Math.cbrt(t) : (7.787 * t + 16 / 116);
    x = f(x); y = f(y); z = f(z);
    return [116 * y - 16, 500 * (x - y), 200 * (y - z)];
  }

  function labToRgb(L, a, b) {
    let y = (L + 16) / 116, x = a / 500 + y, z = y - b / 200;
    const finv = (t) => t > 0.206893 ? t * t * t : (t - 16 / 116) / 7.787;
    x = 0.95047 * finv(x); y = 1.00000 * finv(y); z = 1.08883 * finv(z);
    // XYZ → linear RGB
    let rl = x * 3.2404542 + y * -1.5371385 + z * -0.4985314;
    let gl = x * -0.9692660 + y * 1.8760108 + z * 0.0415560;
    let bl = x * 0.0556434 + y * -0.2040259 + z * 1.0572252;
    // linear → sRGB
    const gamma = (v) => v > 0.0031308 ? 1.055 * Math.pow(v, 1 / 2.4) - 0.055 : 12.92 * v;
    return [
      Math.max(0, Math.min(255, Math.round(gamma(rl) * 255))),
      Math.max(0, Math.min(255, Math.round(gamma(gl) * 255))),
      Math.max(0, Math.min(255, Math.round(gamma(bl) * 255)))
    ];
  }

  /**
   * Reinhard 色彩迁移 + 结构保持
   * 将风格图的色彩统计迁移到内容图上，同时保持内容图的空间结构
   */
  function fallbackColorTransfer(contentCanvas, styleCanvas, strength) {
    strength = strength || 0.85;

    const cw = contentCanvas.width, ch = contentCanvas.height;
    const ctxC = contentCanvas.getContext('2d', { willReadFrequently: true });
    const contentData = ctxC.getImageData(0, 0, cw, ch);

    const sw = styleCanvas.width, sh = styleCanvas.height;
    const ctxS = styleCanvas.getContext('2d', { willReadFrequently: true });
    const styleData = ctxS.getImageData(0, 0, sw, sh);

    // 采样统计（每隔几个像素，降低计算量）
    const step = Math.max(1, Math.floor(Math.sqrt(sw * sh / 5000)));
    let sL = [], sA = [], sB = [];
    for (let i = 0; i < styleData.data.length; i += 4 * step) {
      const lab = rgbToLab(styleData.data[i], styleData.data[i + 1], styleData.data[i + 2]);
      sL.push(lab[0]); sA.push(lab[1]); sB.push(lab[2]);
    }

    // 风格图 LAB 均值和标准差
    const mean = (arr) => arr.reduce((a, b) => a + b, 0) / arr.length;
    const std = (arr, m) => Math.sqrt(arr.reduce((a, b) => a + (b - m) ** 2, 0) / arr.length);
    const sMeanL = mean(sL), sMeanA = mean(sA), sMeanB = mean(sB);
    const sStdL = std(sL, sMeanL) || 1, sStdA = std(sA, sMeanA) || 1, sStdB = std(sB, sMeanB) || 1;

    // 内容图 LAB 统计
    let cL = [], cA = [], cB = [];
    for (let i = 0; i < contentData.data.length; i += 4 * step) {
      const lab = rgbToLab(contentData.data[i], contentData.data[i + 1], contentData.data[i + 2]);
      cL.push(lab[0]); cA.push(lab[1]); cB.push(lab[2]);
    }
    const cMeanL = mean(cL), cMeanA = mean(cA), cMeanB = mean(cB);
    const cStdL = std(cL, cMeanL) || 1, cStdA = std(cA, cMeanA) || 1, cStdB = std(cB, cMeanB) || 1;

    // 对每个像素应用 Reinhard 迁移
    const out = ctxC.createImageData(cw, ch);
    for (let i = 0; i < contentData.data.length; i += 4) {
      const r = contentData.data[i], g = contentData.data[i + 1], b = contentData.data[i + 2];
      const lab = rgbToLab(r, g, b);

      // Reinhard: 将内容图的 LAB 分布映射到风格图的分布
      let nL = ((lab[0] - cMeanL) / cStdL) * sStdL + sMeanL;
      let nA = ((lab[1] - cMeanA) / cStdA) * sStdA + sMeanA;
      let nB = ((lab[2] - cMeanB) / cStdB) * sStdB + sMeanB;

      // 用 strength 控制混合程度
      nL = lab[0] * (1 - strength) + nL * strength;
      nA = lab[1] * (1 - strength) + nA * strength;
      nB = lab[2] * (1 - strength) + nB * strength;

      const rgb = labToRgb(nL, nA, nB);
      out.data[i] = rgb[0];
      out.data[i + 1] = rgb[1];
      out.data[i + 2] = rgb[2];
      out.data[i + 3] = 255;
    }

    return out;
  }

  // 轻量风格迁移（不需要 TF.js，纯 Canvas 操作）
  async function runFallbackStyleTransfer() {
    setStatus('正在使用轻量模式处理…（无需加载模型，秒出结果）');

    const contentImg = await new Promise((res, rej) => {
      const i = new Image(); i.crossOrigin = 'anonymous';
      i.onload = () => res(i); i.onerror = () => rej(new Error('内容图加载失败'));
      i.src = els.contentPreview.src;
    });
    const styleImg = await new Promise((res, rej) => {
      const i = new Image(); i.crossOrigin = 'anonymous';
      i.onload = () => res(i); i.onerror = () => rej(new Error('风格图加载失败'));
      i.src = els.stylePreview.src;
    });

    // 轻量模式下使用较小尺寸
    const lowEnd = isLowEndDevice();
    const maxSide = lowEnd ? 600 : 1024;
    const contentCanvas = downscaleToCanvas(contentImg, maxSide);
    const styleCanvas = downscaleToCanvas(styleImg, Math.min(maxSide, 512));

    // 立即释放 Image
    contentImg.src = ''; styleImg.src = '';

    await new Promise(r => setTimeout(r, 50)); // 让 UI 更新

    // 色彩迁移
    const result = fallbackColorTransfer(contentCanvas, styleCanvas, 0.85);

    // 绘制结果
    els.resultCanvas.width = contentCanvas.width;
    els.resultCanvas.height = contentCanvas.height;
    const ctx = els.resultCanvas.getContext('2d');
    ctx.putImageData(result, 0, 0);

    // 清理
    contentCanvas.width = 0; contentCanvas.height = 0;
    styleCanvas.width = 0; styleCanvas.height = 0;

    resultReady = true;
    els.downloadBtn.disabled = false;
    setStatus('✅ 轻量模式完成！（色彩迁移，无神经网络）可点击"下载结果"。');
  }

  // ============ 核心推理（带自动降级） ============

  async function runStyleTransfer() {
    if (isRunning) return;
    isRunning = true;

    const refs = { contentCanvas: null, styleCanvas: null, contentImg: null, styleImg: null };

    try {
      els.runBtn.disabled = true;
      els.downloadBtn.disabled = true;
      setStatus('处理中…');

      if (!modelReady) {
        if (modelLoading) {
          setStatus('模型正在加载，请稍候…');
        } else {
          // 模型未加载，尝试加载；如果设备太差，直接走轻量模式
          setStatus('正在尝试加载模型…');
          try {
            await withTimeout(ensureModel(), 30000, '模型加载');
          } catch (modelErr) {
            console.warn('模型加载失败，降级到轻量模式：', modelErr);
            try {
              await runFallbackStyleTransfer();
            } catch (_) {
              setStatus('⚠️ 处理失败，请刷新页面重试');
            }
            isRunning = false;
            enableRunIfReady();
            return;
          }
        }
        if (!modelReady) {
          isRunning = false;
          enableRunIfReady();
          return;
        }
      }

      // 推理前内存检查 — 不足时直接降级
      if (!checkMemoryBeforeRun()) {
        setStatus('⚠️ 可用内存不足，切换到轻量模式…');
        try {
          await runFallbackStyleTransfer();
        } catch (_) {
          setStatus('⚠️ 处理失败，请关闭其他标签页后重试');
        }
        isRunning = false;
        enableRunIfReady();
        return;
      }

      const mdl = await ensureModel();

      // 加载 Image 对象
      refs.contentImg = await new Promise((res, rej) => {
        const i = new Image(); i.crossOrigin = 'anonymous';
        i.onload = () => res(i); i.onerror = () => rej(new Error('内容图加载失败'));
        i.src = els.contentPreview.src;
      });
      refs.styleImg = await new Promise((res, rej) => {
        const i = new Image(); i.crossOrigin = 'anonymous';
        i.onload = () => res(i); i.onerror = () => rej(new Error('风格图加载失败'));
        i.src = els.stylePreview.src;
      });

      // 根据后端/设备动态缩放
      const backend = tf.getBackend();
      const sizes = pickSizesByBackend(backend);
      refs.contentCanvas = downscaleToCanvas(refs.contentImg, sizes.content);
      refs.styleCanvas = downscaleToCanvas(refs.styleImg, sizes.style);

      // ★ 缩放完成后立即释放 Image 引用
      refs.contentImg.src = '';
      refs.styleImg.src = '';
      refs.contentImg = null;
      refs.styleImg = null;

      const lowEnd = isLowEndDevice();
      const inferTimeout = lowEnd ? 90000 : 180000;
      setStatus('推理中…（' + (lowEnd ? '设备性能较低，请耐心等待' : '取决于设备性能，可能需数秒到十数秒') + '）');

      tf.engine().startScope();
      let stylized;
      try {
        stylized = await withRetry(() => mdl.stylize(refs.contentCanvas, refs.styleCanvas), {
          retries: lowEnd ? 0 : 1,
          baseDelay: 600,
          timeoutMs: inferTimeout,
          onAttempt: (k) => setStatus(k ? '重试推理中…' : '推理中…')
        });
      } catch (inferErr) {
        tf.engine().endScope();
        throw inferErr;
      }

      const ctx = els.resultCanvas.getContext('2d');

      // 处理不同返回类型
      if (typeof tf !== 'undefined' && tf.tensor && stylized && typeof stylized === 'object' && typeof stylized.data === 'function' && Array.isArray(stylized.shape)) {
        let t = stylized;
        let squeezed = false;
        if (t.shape.length === 4 && t.shape[0] === 1) {
          t = t.squeeze();
          squeezed = true;
        }
        if (t.shape.length !== 3) {
          t.dispose();
          throw new Error('模型返回张量维度不支持：' + t.shape.join('x'));
        }
        const [h, w] = t.shape.slice(0, 2);
        els.resultCanvas.width = w;
        els.resultCanvas.height = h;
        const data = await t.data();
        // ★ 关键修复：提取数据后立即释放张量
        t.dispose();
        if (squeezed && stylized !== t) stylized.dispose();

        const imageData = ctx.createImageData(w, h);
        const scale = (function () {
          let maxv = 0, minv = 1e9;
          const sampleLen = Math.min(1000, data.length);
          for (let i = 0; i < sampleLen; i++) {
            const v = data[i]; maxv = Math.max(maxv, v); minv = Math.min(minv, v);
          }
          return (maxv <= 1.5) ? 255 : 1;
        })();
        for (let i = 0, j = 0; i < data.length; i += 3, j += 4) {
          imageData.data[j] = data[i] * scale;
          imageData.data[j + 1] = data[i + 1] * scale;
          imageData.data[j + 2] = data[i + 2] * scale;
          imageData.data[j + 3] = 255;
        }
        ctx.putImageData(imageData, 0, 0);
      } else if (stylized instanceof HTMLCanvasElement) {
        els.resultCanvas.width = stylized.width;
        els.resultCanvas.height = stylized.height;
        ctx.drawImage(stylized, 0, 0);
      } else if (stylized && typeof stylized.width === 'number' && typeof stylized.height === 'number' && stylized.data) {
        els.resultCanvas.width = stylized.width;
        els.resultCanvas.height = stylized.height;
        ctx.putImageData(stylized, 0, 0);
      } else {
        throw new Error('模型返回未知类型，无法绘制');
      }
      tf.engine().endScope();

      resultReady = true;
      els.downloadBtn.disabled = false;
      setStatus('完成！可点击"下载结果"。（后端：' + tf.getBackend() + '）');
    } catch (err) {
      console.error('风格迁移错误：', err);
      let msg = err && err.message ? err.message : String(err);
      const isOom = msg.includes('out of memory') || msg.includes('OOM');
      const isTimeout = msg.includes('超时');
      const isWebGL = msg.includes('WebGL');

      // 自动降级到轻量模式
      if (isOom || isTimeout || isWebGL) {
        console.warn('神经网络推理失败，自动降级到轻量色彩迁移模式');
        try {
          await runFallbackStyleTransfer();
          return; // 轻量模式成功，直接返回
        } catch (fallbackErr) {
          console.error('轻量模式也失败了：', fallbackErr);
          setStatus('⚠️ 两种模式均失败，请刷新页面重试');
          return;
        }
      }

      if (isOom) {
        msg = '⚠️ 内存不足！正在尝试轻量模式…';
      } else if (isTimeout) {
        msg = '⚠️ ' + msg + '。正在尝试轻量模式…';
      } else if (isWebGL) {
        msg = '⚠️ WebGL 不可用，正在尝试轻量模式…';
      }
      setStatus('发生错误：' + msg);
    } finally {
      // ★ 关键修复：无论如何都清理临时资源
      cleanupInferenceResources(refs);
      isRunning = false;
      enableRunIfReady();
    }
  }

  // ============ 文件选择 & 下载 ============

  function onFileChange(which) {
    return async (e) => {
      const file = e.target.files && e.target.files[0];
      if (!file) return;
      try {
        await fileToImage(file);
        const url = URL.createObjectURL(file);
        if (which === 'content') {
          els.contentPreview.src = url;
          const box = els.contentPreview.closest('.preview');
          if (box) box.classList.remove('empty');
        } else {
          els.stylePreview.src = url;
          const box = els.stylePreview.closest('.preview');
          if (box) box.classList.remove('empty');
        }
        resultReady = false;
        els.downloadBtn.disabled = true;
        setStatus('已选择图片');
      } catch (e) {
        console.error(e);
        setStatus('读取图片失败');
      } finally {
        enableRunIfReady();
      }
    };
  }

  function downloadResult() {
    if (!resultReady) return;
    const link = document.createElement('a');
    link.download = 'stylized.png';
    link.href = els.resultCanvas.toDataURL('image/png');
    link.click();
  }

  // ============ 初始化 ============

  function setup() {
    els.contentInput.addEventListener('change', onFileChange('content'));
    els.styleInput.addEventListener('change', onFileChange('style'));
    els.runBtn.addEventListener('click', runStyleTransfer);
    els.fallbackBtn.addEventListener('click', async function() {
      if (isRunning) return;
      isRunning = true;
      els.runBtn.disabled = true;
      els.fallbackBtn.disabled = true;
      els.downloadBtn.disabled = true;
      try {
        await runFallbackStyleTransfer();
      } catch (err) {
        console.error('轻量模式错误：', err);
        setStatus('轻量模式失败：' + (err.message || err));
      } finally {
        isRunning = false;
        enableRunIfReady();
      }
    });
    els.downloadBtn.addEventListener('click', downloadResult);

    // 渲染示例图库
    (function renderSamples() {
      var contentBox = document.getElementById('sampleContent');
      var styleBox = document.getElementById('sampleStyle');
      if (!contentBox && !styleBox) return;
      var CONTENT_SAMPLES = [
        './img_content/cat.jpg',
        './img_content/dog.jpg',
        './img_content/flower.jpg',
        './img_content/xian_bell_tower.jpg'
      ];
      var STYLE_SAMPLES = [
        './img_style/Monet.png',
        './img_style/Linear.jpg',
        './img_style/Vincent_starrynight.jpg',
        './img_style/BG_landscape_painting.jpg'
      ];
      function createCard(url, which) {
        var card = document.createElement('div');
        card.className = 'sample-card';
        var img = document.createElement('img');
        img.src = url;
        img.alt = url.split('/').pop();
        img.onerror = function () { card.remove(); };
        var actions = document.createElement('div');
        actions.className = 'sample-actions';
        var btn = document.createElement('button');
        btn.setAttribute('data-which', which);
        btn.setAttribute('data-url', url);
        btn.textContent = which === 'content' ? '作为内容' : '作为风格';
        actions.appendChild(btn);
        card.appendChild(img);
        card.appendChild(actions);
        return card;
      }
      if (contentBox) {
        CONTENT_SAMPLES.forEach(function (u) { contentBox.appendChild(createCard(u, 'content')); });
      }
      if (styleBox) {
        STYLE_SAMPLES.forEach(function (u) { styleBox.appendChild(createCard(u, 'style')); });
      }
      document.addEventListener('click', function (e) {
        var t = e.target;
        if (t && t.matches('.sample-actions button')) {
          var which = t.getAttribute('data-which');
          var url = t.getAttribute('data-url');
          try {
            if (which === 'content') {
              els.contentPreview.crossOrigin = 'anonymous';
              els.contentPreview.src = url;
              var boxC = els.contentPreview.closest('.preview');
              if (boxC) boxC.classList.remove('empty');
            } else {
              els.stylePreview.crossOrigin = 'anonymous';
              els.stylePreview.src = url;
              var boxS = els.stylePreview.closest('.preview');
              if (boxS) boxS.classList.remove('empty');
            }
            setStatus('已选择示例图片');
            enableRunIfReady();
          } catch (_) { }
        }
      });
    })();

    // 若从首页选择了示例，自动填充预览
    try {
      var presetContent = localStorage.getItem('stylemigrate_preset_content');
      var presetStyle = localStorage.getItem('stylemigrate_preset_style');
      if (presetContent) {
        els.contentPreview.crossOrigin = 'anonymous';
        els.contentPreview.src = presetContent;
        var boxC = els.contentPreview.closest('.preview');
        if (boxC) boxC.classList.remove('empty');
      }
      if (presetStyle) {
        els.stylePreview.crossOrigin = 'anonymous';
        els.stylePreview.src = presetStyle;
        var boxS = els.stylePreview.closest('.preview');
        if (boxS) boxS.classList.remove('empty');
      }
      if (presetContent || presetStyle) {
        setStatus('已从示例填充图片');
        enableRunIfReady();
        localStorage.removeItem('stylemigrate_preset_content');
        localStorage.removeItem('stylemigrate_preset_style');
      }
    } catch (_) { }

    // ★ 优化：延迟 3 秒预热模型，不与页面渲染竞争
    setTimeout(async () => {
      try {
        await ensureModel();
      } catch (e) {
        console.warn('模型预加载失败（不影响正常使用）', e);
        modelLoading = false;
        setStatus('模型预加载失败，请点击"开始风格迁移"手动加载');
      }
    }, 3000);
  }

  document.addEventListener('DOMContentLoaded', setup);
})();
