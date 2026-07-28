/**
 * 图片加载增强模块
 * - 自动重试（网络抖动时 3 次重试，指数退避）
 * - 加载失败占位图（SVG 内联，零网络请求）
 * - 渐显动画（加载完成后淡入）
 * - lazy loading（非首屏图片延迟加载）
 */
(function () {
  'use strict';

  var MAX_RETRIES = 3;
  var BASE_DELAY = 800; // 首次重试延迟 ms

  // ── 失败占位图（内联 SVG，无需网络请求） ──
  var PLACEHOLDER_SVG = 'data:image/svg+xml,' + encodeURIComponent(
    '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="300" viewBox="0 0 400 300">' +
    '<rect fill="#f1f5f9" width="400" height="300"/>' +
    '<text x="200" y="140" text-anchor="middle" fill="#94a3b8" font-family="sans-serif" font-size="14">图片加载失败</text>' +
    '<text x="200" y="165" text-anchor="middle" fill="#cbd5e1" font-family="sans-serif" font-size="12">点击重试</text>' +
    '<g transform="translate(188,180)" fill="none" stroke="#94a3b8" stroke-width="2" stroke-linecap="round">' +
    '<path d="M1 12a11 11 0 1 0 3-8"/>' +
    '<polyline points="1 1 1 5 5 5"/>' +
    '</g></svg>'
  );

  // ── 注入样式 ──
  var style = document.createElement('style');
  style.textContent =
    '.img-fade-in{opacity:0;transition:opacity .4s ease}' +
    '.img-fade-in.img-loaded{opacity:1}' +
    '.img-fade-in.img-error{opacity:1;cursor:pointer;filter:grayscale(.2)}';
  (document.head || document.documentElement).appendChild(style);

  // ── 核心：为单个 img 元素应用增强 ──
  function enhance(img) {
    // 跳过已处理或 data: 图片
    if (img.dataset.enhanced) return;
    if (img.src && img.src.indexOf('data:') === 0 && img.src.indexOf('svg+xml') === -1) return;
    img.dataset.enhanced = '1';

    // 1) lazy loading（非首屏）
    if (!img.hasAttribute('loading') && !isInViewport(img)) {
      img.setAttribute('loading', 'lazy');
    }

    // 2) 渐显动画
    if (!img.classList.contains('img-fade-in')) {
      img.classList.add('img-fade-in');
    }

    // 3) 如果已经加载完成（缓存命中）
    if (img.complete && img.naturalWidth > 0) {
      img.classList.add('img-loaded');
      return;
    }

    var retryCount = 0;
    var originalSrc = img.src;

    // 加载成功
    function onSuccess() {
      img.classList.add('img-loaded');
      img.classList.remove('img-error');
      img.removeEventListener('error', onError);
    }

    // 加载失败 → 自动重试
    function onError() {
      img.removeEventListener('load', onSuccess);
      if (retryCount < MAX_RETRIES) {
        retryCount++;
        var delay = BASE_DELAY * Math.pow(2, retryCount - 1);
        setTimeout(function () {
          // 给 URL 加时间戳绕过缓存
          var sep = originalSrc.indexOf('?') > -1 ? '&' : '?';
          img.src = originalSrc + sep + '_retry=' + Date.now() + '_' + retryCount;
        }, delay);
      } else {
        // 最终失败 → 显示占位图
        img.src = PLACEHOLDER_SVG;
        img.classList.add('img-error');
        img.title = '图片加载失败，点击重试';
        img.style.cursor = 'pointer';
        // 点击占位图重试
        img.addEventListener('click', function retryClick() {
          retryCount = 0;
          img.removeEventListener('click', retryClick);
          img.style.cursor = '';
          img.title = '';
          img.classList.remove('img-error');
          img.src = originalSrc;
          bindEvents();
        });
      }
    }

    function bindEvents() {
      img.addEventListener('load', onSuccess, { once: true });
      img.addEventListener('error', onError);
    }

    bindEvents();
  }

  // ── 判断元素是否在视口内 ──
  function isInViewport(el) {
    try {
      var rect = el.getBoundingClientRect();
      return rect.top < window.innerHeight + 200 && rect.bottom > -200 &&
             rect.left < window.innerWidth + 200 && rect.right > -200;
    } catch (_) { return true; }
  }

  // ── 增强所有现有 img ──
  function enhanceAll() {
    var imgs = document.querySelectorAll('img:not([data-enhanced])');
    for (var i = 0; i < imgs.length; i++) {
      enhance(imgs[i]);
    }
  }

  // ── 监听动态插入的 img（MutationObserver） ──
  function observe() {
    var observer = new MutationObserver(function (mutations) {
      for (var i = 0; i < mutations.length; i++) {
        var added = mutations[i].addedNodes;
        for (var j = 0; j < added.length; j++) {
          var node = added[j];
          if (node.nodeName === 'IMG') {
            enhance(node);
          } else if (node.querySelectorAll) {
            var imgs = node.querySelectorAll('img:not([data-enhanced])');
            for (var k = 0; k < imgs.length; k++) {
              enhance(imgs[k]);
            }
          }
        }
      }
    });
    observer.observe(document.body || document.documentElement, { childList: true, subtree: true });
  }

  // ── 初始化 ──
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () { enhanceAll(); observe(); });
  } else {
    enhanceAll();
    observe();
  }
})();
