/**
 * 全局水印模块
 * 斜体白色中文黑体字水印，4x4网格，50%不透明度
 */
(function () {
  const WATERMARK_TEXT = '西电高代课程组';
  const ROWS = 4;
  const COLS = 4;
  const FONT_SIZE = 18;
  const ROTATE_DEG = -30;
  const OPACITY = 0.5;
  const FONT_FAMILY = '"SimHei", "黑体", "Microsoft YaHei", sans-serif';

  function createWatermark() {
    // 移除旧水印（防止重复）
    const old = document.getElementById('global-watermark');
    if (old) old.remove();

    const container = document.createElement('div');
    container.id = 'global-watermark';
    container.style.cssText = `
      position: fixed;
      top: 0; left: 0;
      width: 100vw; height: 100vh;
      pointer-events: none;
      z-index: 999999;
      overflow: hidden;
    `;

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    // 计算单个水印单元的尺寸
    const textWidth = ctx.measureText(WATERMARK_TEXT).width || 150;
    const cellW = Math.max(textWidth + 80, 200);
    const cellH = 120;
    canvas.width = cellW * COLS;
    canvas.height = cellH * ROWS;

    ctx.font = `${FONT_SIZE}px ${FONT_FAMILY}`;
    ctx.fillStyle = 'rgba(255, 255, 255, ' + OPACITY + ')';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        const x = cellW * c + cellW / 2;
        const y = cellH * r + cellH / 2;
        ctx.save();
        ctx.translate(x, y);
        ctx.rotate((ROTATE_DEG * Math.PI) / 180);
        ctx.fillText(WATERMARK_TEXT, 0, 0);
        ctx.restore();
      }
    }

    container.style.backgroundImage = `url(${canvas.toDataURL('image/png')})`;
    container.style.backgroundRepeat = 'repeat';
    container.style.backgroundSize = `${canvas.width}px ${canvas.height}px`;

    document.body.appendChild(container);
  }

  // 页面加载后创建水印
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', createWatermark);
  } else {
    createWatermark();
  }

  // 监听 DOM 变化，防止水印被移除
  const observer = new MutationObserver(function () {
    if (!document.getElementById('global-watermark')) {
      createWatermark();
    }
  });
  observer.observe(document.body, { childList: true, subtree: true });
})();
