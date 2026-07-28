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
  let modelLoadPromise = null; // 缓存加载 Promise，防止并发重复加载
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

  // ── 设备探测 ──
  function getDeviceProfile() {
    var mem = 0, cores = 2, gpu = '', webgl = false;
    try { mem = Number(navigator.deviceMemory || 0); } catch(_){}
    try { cores = navigator.hardwareConcurrency || 2; } catch(_){}
    try {
      var c = document.createElement('canvas');
      var gl = c.getContext('webgl2') || c.getContext('webgl') || c.getContext('experimental-webgl');
      if (gl) {
        webgl = true;
        var ext = gl.getExtension('WEBGL_debug_renderer_info');
        if (ext) gpu = gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) || '';
      }
    } catch(_){}
    var tier;
    if (!webgl || (mem > 0 && mem <= 2) || cores <= 1) tier = 'low';
    else if ((mem > 0 && mem <= 4) || cores <= 2) tier = 'mid';
    else tier = 'high';
    return { mem: mem, cores: cores, gpu: gpu, webgl: webgl, tier: tier };
  }

  var deviceProfile = getDeviceProfile();
  console.log('[设备] tier=' + deviceProfile.tier + ' mem=' + deviceProfile.mem + 'GB cores=' + deviceProfile.cores + ' webgl=' + deviceProfile.webgl + ' gpu=' + deviceProfile.gpu);

  function isLowEndDevice() { return deviceProfile.tier === 'low'; }

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
    // 如果正在加载，等待现有加载完成，不重复启动
    if (modelLoadPromise) return modelLoadPromise;
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
          model = null; modelReady = false; modelLoadPromise = null;
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
    // 缓存 Promise，防止并发重复加载
    modelLoadPromise = (async () => {
      try {
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
        setStatus('模型就绪（后端：' + backend + '）');
        return model;
      } finally {
        modelLoading = false;
        modelLoadPromise = null;
      }
    })();
    return modelLoadPromise;
  }

  // ============ 优化模型（MobileNet-v2 + 可分离卷积，~12MB） ============

  let optStyleNet = null;
  let optTransformerNet = null;
  let optModelsReady = false;
  let optLoadPromise = null;

  async function loadOptimizedModels() {
    if (optModelsReady) return;
    if (optLoadPromise) return optLoadPromise;

    if (!(window.tf && window.tf.engine)) throw new Error('TensorFlow.js 未加载');

    optLoadPromise = (async function() {
      // 确保后端就绪
      try { await tf.setBackend('webgl'); } catch (_) {}
      try { await tf.setBackend('wasm'); } catch (_) {}
      try { await tf.setBackend('cpu'); } catch (_) {}
      await tf.ready();
      console.log('[优化模型] 后端:', tf.getBackend());

      var BASE = 'https://cdn.jsdelivr.net/gh/reiinakano/arbitrary-image-stylization-tfjs@master';

      setStatus('加载风格网络…');
      console.log('[优化模型] 加载风格网络…');
      optStyleNet = await tf.loadGraphModel(BASE + '/saved_model_style_js/model.json');
      console.log('[优化模型] 风格网络完成');

      setStatus('加载转换网络…（较大，请耐心等待）');
      console.log('[优化模型] 加载转换网络…');
      optTransformerNet = await tf.loadGraphModel(BASE + '/saved_model_transformer_separable_js/model.json');
      console.log('[优化模型] 转换网络完成');

      optModelsReady = true;
      setStatus('优化模型就绪');
    })();

    try {
      await optLoadPromise;
    } finally {
      optLoadPromise = null;
    }
  }

  async function runFallbackStyleTransfer() {
    await loadOptimizedModels();

    setStatus('正在使用优化模型处理…');

    var contentImg = await new Promise(function (res, rej) {
      var i = new Image(); i.crossOrigin = 'anonymous';
      i.onload = function () { res(i); }; i.onerror = function () { rej(new Error('内容图加载失败')); };
      i.src = els.contentPreview.src;
    });
    var styleImg = await new Promise(function (res, rej) {
      var i = new Image(); i.crossOrigin = 'anonymous';
      i.onload = function () { res(i); }; i.onerror = function () { rej(new Error('风格图加载失败')); };
      i.src = els.stylePreview.src;
    });

    var lowEnd = isLowEndDevice();
    var contentCanvas = downscaleToCanvas(contentImg, lowEnd ? 384 : 640);
    var styleCanvas = downscaleToCanvas(styleImg, lowEnd ? 192 : 384);
    contentImg.src = ''; styleImg.src = '';

    tf.engine().startScope();
    try {
      var content = tf.browser.fromPixels(contentCanvas, 3).toFloat().div(255).expandDims();
      var style = tf.browser.fromPixels(styleCanvas, 3).toFloat().div(255).expandDims();
      var styleVec = optStyleNet.predict(style);
      var result = optTransformerNet.predict([content, styleVec]);
      var squeezed = result.squeeze();
      els.resultCanvas.width = squeezed.shape[1];
      els.resultCanvas.height = squeezed.shape[0];
      await tf.browser.toPixels(squeezed, els.resultCanvas);
      squeezed.dispose(); styleVec.dispose(); content.dispose(); style.dispose();
      tf.engine().endScope();
      contentCanvas.width = 0; contentCanvas.height = 0;
      styleCanvas.width = 0; styleCanvas.height = 0;
      resultReady = true;
      els.downloadBtn.disabled = false;
      setStatus('✅ 完成！可点击"下载结果"。（优化模型，后端：' + tf.getBackend() + '）');
    } catch (e) {
      tf.engine().endScope();
      throw new Error('推理失败：' + e.message);
    }
  }

  // ============ 核心推理（设备自适应） ============

  async function runStyleTransfer() {
    if (isRunning) return;
    isRunning = true;

    var refs = { contentCanvas: null, styleCanvas: null, contentImg: null, styleImg: null };

    try {
      els.runBtn.disabled = true;
      els.downloadBtn.disabled = true;

      // 设备自适应：high 用完整模型，mid/low 用优化模型
      var useOptModel = deviceProfile.tier !== 'high';

      if (useOptModel) {
        // ── 优化模型路径 ──
        setStatus('正在加载优化模型…');
        await loadOptimizedModels();
        setStatus('正在处理…');
        await runFallbackStyleTransfer();
        return;
      }

      // ── 完整模型路径（仅 high tier） ──
      if (!modelReady) {
        setStatus('正在加载完整模型，请稍候…');
        try {
          await withTimeout(ensureModel(), 120000, '模型加载');
        } catch (modelErr) {
          console.warn('完整模型加载失败，降级到优化模型：', modelErr);
          await loadOptimizedModels();
          await runFallbackStyleTransfer();
          return;
        }
        if (!modelReady) {
          isRunning = false;
          enableRunIfReady();
          return;
        }
      }

      if (!checkMemoryBeforeRun()) {
        setStatus('⚠️ 可用内存不足，使用优化模型…');
        await loadOptimizedModels();
        await runFallbackStyleTransfer();
        return;
      }

      var mdl = await ensureModel();

      refs.contentImg = await new Promise(function (res, rej) {
        var i = new Image(); i.crossOrigin = 'anonymous';
        i.onload = function () { res(i); }; i.onerror = function () { rej(new Error('内容图加载失败')); };
        i.src = els.contentPreview.src;
      });
      refs.styleImg = await new Promise(function (res, rej) {
        var i = new Image(); i.crossOrigin = 'anonymous';
        i.onload = function () { res(i); }; i.onerror = function () { rej(new Error('风格图加载失败')); };
        i.src = els.stylePreview.src;
      });

      var backend = tf.getBackend();
      var sizes = pickSizesByBackend(backend);
      refs.contentCanvas = downscaleToCanvas(refs.contentImg, sizes.content);
      refs.styleCanvas = downscaleToCanvas(refs.styleImg, sizes.style);
      refs.contentImg.src = ''; refs.styleImg.src = '';
      refs.contentImg = null; refs.styleImg = null;

      setStatus('推理中…');

      tf.engine().startScope();
      var stylized;
      try {
        stylized = await withRetry(function () { return mdl.stylize(refs.contentCanvas, refs.styleCanvas); }, {
          retries: 0,
          baseDelay: 600,
          timeoutMs: 180000
        });
      } catch (inferErr) {
        tf.engine().endScope();
        throw inferErr;
      }

      var ctx = els.resultCanvas.getContext('2d');

      if (stylized && typeof stylized.data === 'function' && Array.isArray(stylized.shape)) {
        var t = stylized;
        var squeezed = false;
        if (t.shape.length === 4 && t.shape[0] === 1) { t = t.squeeze(); squeezed = true; }
        if (t.shape.length !== 3) { t.dispose(); throw new Error('张量维度不支持：' + t.shape.join('x')); }
        var h = t.shape[0], w = t.shape[1];
        els.resultCanvas.width = w; els.resultCanvas.height = h;
        var data = await t.data();
        t.dispose();
        if (squeezed && stylized !== t) stylized.dispose();
        var imageData = ctx.createImageData(w, h);
        var scale = 1;
        for (var si = 0; si < Math.min(1000, data.length); si++) { if (data[si] > 1.5) { scale = 1; break; } scale = 255; }
        for (var di = 0, dj = 0; di < data.length; di += 3, dj += 4) {
          imageData.data[dj] = data[di] * scale;
          imageData.data[dj + 1] = data[di + 1] * scale;
          imageData.data[dj + 2] = data[di + 2] * scale;
          imageData.data[dj + 3] = 255;
        }
        ctx.putImageData(imageData, 0, 0);
      } else if (stylized instanceof HTMLCanvasElement) {
        els.resultCanvas.width = stylized.width; els.resultCanvas.height = stylized.height;
        ctx.drawImage(stylized, 0, 0);
      } else {
        throw new Error('模型返回未知类型');
      }
      tf.engine().endScope();

      resultReady = true;
      els.downloadBtn.disabled = false;
      setStatus('✅ 完成！可点击"下载结果"。（完整模型，后端：' + tf.getBackend() + '）');
    } catch (err) {
      console.error('风格迁移错误：', err);
      var msg = err && err.message ? err.message : String(err);
      // 完整模型失败 → 降级到优化模型
      if (deviceProfile.tier === 'high') {
        console.warn('完整模型失败，降级到优化模型');
        try {
          await loadOptimizedModels();
          await runFallbackStyleTransfer();
          return;
        } catch (_) {
          setStatus('⚠️ 处理失败：' + msg);
          return;
        }
      }
      setStatus('发生错误：' + msg);
    } finally {
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
