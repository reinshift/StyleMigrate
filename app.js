/*
  StyleMigrate - 前端任意风格迁移（TensorFlow.js + @magenta/image）
  说明：完全在浏览器本地运行，不上传图片到服务器。
*/

(() => {
  const els = {
    contentInput: document.getElementById('contentInput'),
    styleInput: document.getElementById('styleInput'),
    contentPreview: document.getElementById('contentPreview'),
    stylePreview: document.getElementById('stylePreview'),
    runBtn: document.getElementById('runBtn'),
    downloadBtn: document.getElementById('downloadBtn'),
    status: document.getElementById('status'),
    resultCanvas: document.getElementById('resultCanvas'),
  };

  const MAX_SIDE = 1024; // 限制最长边，平衡速度/内存
  let model = null; // mi.ArbitraryStyleTransferNetwork 实例
  let resultReady = false;

  function setStatus(text) {
    els.status.textContent = text;
  }

  function enableRunIfReady() {
    const hasContent = !!els.contentPreview.src;
    const hasStyle = !!els.stylePreview.src;
    els.runBtn.disabled = !(hasContent && hasStyle);
    setStatus(hasContent && hasStyle ? '点击“开始风格迁移”。' : '请先选择内容图与风格图。');
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

  async function ensureModel() {
    if (model) return model;
    if (!(window.tf && window.tf.engine)) {
      throw new Error('TensorFlow.js 未正确加载');
    }
    if (!(window.mi && window.mi.ArbitraryStyleTransferNetwork)) {
      throw new Error('@magenta/image 未正确加载');
    }
    // WebGL 检查
    const backend = tf.getBackend();
    if (backend !== 'webgl' && backend !== 'webgpu') {
      try { await tf.setBackend('webgl'); await tf.ready(); }
      catch (_) {}
    }
    setStatus('正在加载模型…（首次需要数秒）');
    model = new mi.ArbitraryStyleTransferNetwork();
    if (typeof model.initialize === 'function') {
      await model.initialize();
    }
    setStatus('模型就绪');
    return model;
  }

  async function runStyleTransfer() {
    try {
      els.runBtn.disabled = true;
      els.downloadBtn.disabled = true;
      setStatus('准备中…');

      const mdl = await ensureModel();

      // 将预览图再次加载为 Image 对象（更安全地得到尺寸）
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

      // 缩放至合适大小
      const contentCanvas = downscaleToCanvas(contentImg, MAX_SIDE);
      const styleCanvas = downscaleToCanvas(styleImg, 512); // 风格图可适当更小

      setStatus('推理中…（取决于设备性能，可能需数秒到十数秒）');

      tf.engine().startScope();
      const stylized = await mdl.stylize(contentCanvas, styleCanvas);
      const ctx = els.resultCanvas.getContext('2d');

      // 兼容不同返回类型
      if (typeof tf !== 'undefined' && tf.tensor && stylized && typeof stylized === 'object' && typeof stylized.data === 'function' && Array.isArray(stylized.shape)) {
        // Tensor（3D 或 4D）
        let t = stylized;
        if (t.shape.length === 4 && t.shape[0] === 1) {
          t = t.squeeze(); // [1,h,w,3] -> [h,w,3]
        }
        if (t.shape.length !== 3) {
          throw new Error('模型返回张量维度不支持：' + t.shape.join('x'));
        }
        const [h, w] = t.shape.slice(0, 2);
        els.resultCanvas.width = w;
        els.resultCanvas.height = h;
        const data = await t.data();
        const imageData = ctx.createImageData(w, h);
        const scale = (function(){
          let maxv = 0, minv = 1e9;
          for (let i = 0; i < Math.min(1000, data.length); i++) {
            const v = data[i]; maxv = Math.max(maxv, v); minv = Math.min(minv, v);
          }
          return (maxv <= 1.5) ? 255 : 1; // 0..1 -> *255
        })();
        for (let i = 0, j = 0; i < data.length; i += 3, j += 4) {
          imageData.data[j] = data[i] * scale;
          imageData.data[j + 1] = data[i + 1] * scale;
          imageData.data[j + 2] = data[i + 2] * scale;
          imageData.data[j + 3] = 255;
        }
        ctx.putImageData(imageData, 0, 0);
      } else if (stylized instanceof HTMLCanvasElement) {
        // 直接是 Canvas
        els.resultCanvas.width = stylized.width;
        els.resultCanvas.height = stylized.height;
        ctx.drawImage(stylized, 0, 0);
      } else if (stylized && typeof stylized.width === 'number' && typeof stylized.height === 'number' && stylized.data) {
        // ImageData
        els.resultCanvas.width = stylized.width;
        els.resultCanvas.height = stylized.height;
        ctx.putImageData(stylized, 0, 0);
      } else {
        throw new Error('模型返回未知类型，无法绘制');
      }
      tf.engine().endScope();

      resultReady = true;
      els.downloadBtn.disabled = false;
      setStatus('完成！可点击“下载结果”。');
    } catch (err) {
      console.error(err);
      setStatus('发生错误：' + (err && err.message ? err.message : err));
    } finally {
      enableRunIfReady();
    }
  }

  function onFileChange(which) {
    return async (e) => {
      const file = e.target.files && e.target.files[0];
      if (!file) return;
      try {
        const img = await fileToImage(file);
        // 预览图不做缩放，直接展示（CSS 限制尺寸）
        const url = URL.createObjectURL(file);
        if (which === 'content') {
          els.contentPreview.src = url;
        } else {
          els.stylePreview.src = url;
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

  function setup() {
    // 事件绑定
    els.contentInput.addEventListener('change', onFileChange('content'));
    els.styleInput.addEventListener('change', onFileChange('style'));
    els.runBtn.addEventListener('click', runStyleTransfer);
    els.downloadBtn.addEventListener('click', downloadResult);

    // 后台预热（可选）
    setTimeout(async () => {
      try {
        await ensureModel();
      } catch (e) {
        console.warn('模型预加载失败', e);
      }
    }, 100);
  }

  document.addEventListener('DOMContentLoaded', setup);
})();
