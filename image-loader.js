/**
 * 图片加载增强模块 v2
 * - 自动重试（网络抖动 3 次，指数退避）
 * - 失败占位图（内联 SVG，零网络请求，点击可重试）
 * - 渐显动画（加载完成后淡入 0.4s）
 * - lazy loading（非首屏图片延迟加载）
 * - 动态监听（MutationObserver 自动增强新插入的图片）
 */
;(function () {
  'use strict';

  var MAX_RETRIES = 3;
  var BASE_DELAY = 800;

  // ── 内联 SVG 占位图 ──
  var PLACEHOLDER = 'data:image/svg+xml,' + encodeURIComponent(
    '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="300" viewBox="0 0 400 300">' +
    '<rect fill="#f1f5f9" width="400" height="300"/>' +
    '<text x="200" y="140" text-anchor="middle" fill="#94a3b8" font-family="sans-serif" font-size="14">\u56FE\u7247\u52A0\u8F7D\u5931\u8D25</text>' +
    '<text x="200" y="165" text-anchor="middle" fill="#cbd5e1" font-family="sans-serif" font-size="12">\u70B9\u51FB\u91CD\u8BD5</text>' +
    '<g transform="translate(188,180)" fill="none" stroke="#94a3b8" stroke-width="2" stroke-linecap="round">' +
    '<path d="M1 12a11 11 0 1 0 3-8"/><polyline points="1 1 1 5 5 5"/></g></svg>'
  );

  // ── 注入渐显 CSS ──
  var css = document.createElement('style');
  css.textContent =
    '.img-enhanced{transition:opacity .4s ease}' +
    '.img-enhanced:not(.img-ok){opacity:0}' +
    '.img-enhanced.img-ok{opacity:1}' +
    '.img-enhanced.img-err{opacity:1;cursor:pointer;filter:grayscale(.15)}';
  (document.head || document.documentElement).appendChild(css);

  // ── 增强单个 <img> ──
  function enhance(img) {
    if (img.dataset.ie) return;
    if (!img.src || img.src.indexOf('data:image/svg+xml') > -1) return;
    img.dataset.ie = '1';

    // 非首屏 → lazy loading
    if (!img.hasAttribute('loading') && !inView(img)) {
      img.setAttribute('loading', 'lazy');
    }

    // 渐显 class
    img.classList.add('img-enhanced');

    // 已缓存加载完成
    if (img.complete && img.naturalWidth > 0) {
      img.classList.add('img-ok');
      return;
    }

    var tries = 0, orig = img.src;

    function onLoad() {
      img.classList.add('img-ok');
      img.classList.remove('img-err');
      img.removeEventListener('error', onErr);
    }

    function onErr() {
      img.removeEventListener('load', onLoad);
      if (tries < MAX_RETRIES) {
        tries++;
        setTimeout(function () {
          img.src = orig + (orig.indexOf('?') > -1 ? '&' : '?') + '_r=' + tries + '&' + Date.now();
        }, BASE_DELAY * Math.pow(2, tries - 1));
      } else {
        img.removeEventListener('error', onErr);
        img.src = PLACEHOLDER;
        img.classList.add('img-err');
        img.title = '\u56FE\u7247\u52A0\u8F7D\u5931\u8D25\uFF0C\u70B9\u51FB\u91CD\u8BD5';
        img.onclick = function () {
          tries = 0;
          img.classList.remove('img-err');
          img.title = '';
          img.src = orig;
          bind();
        };
      }
    }

    function bind() {
      img.addEventListener('load', onLoad, { once: true });
      img.addEventListener('error', onErr);
    }
    bind();
  }

  function inView(el) {
    try {
      var r = el.getBoundingClientRect();
      return r.top < innerHeight + 200 && r.bottom > -200;
    } catch (_) { return true; }
  }

  function scanAll() {
    var imgs = document.querySelectorAll('img:not([data-ie])');
    for (var i = 0; i < imgs.length; i++) enhance(imgs[i]);
  }

  // 立即扫描（脚本在 body 末尾，DOM 已就绪）
  scanAll();

  // 动态插入的图片
  if (typeof MutationObserver !== 'undefined') {
    var mo = new MutationObserver(function (muts) {
      for (var i = 0; i < muts.length; i++) {
        var nodes = muts[i].addedNodes;
        for (var j = 0; j < nodes.length; j++) {
          var n = nodes[j];
          if (n.nodeType !== 1) continue;
          if (n.tagName === 'IMG') enhance(n);
          else if (n.querySelectorAll) {
            var list = n.querySelectorAll('img:not([data-ie])');
            for (var k = 0; k < list.length; k++) enhance(list[k]);
          }
        }
      }
    });
    mo.observe(document.documentElement, { childList: true, subtree: true });
  }

  // 页面完全加载后再扫一遍（兜底慢速加载的图片）
  window.addEventListener('load', scanAll);
})();
