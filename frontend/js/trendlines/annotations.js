/**
 * Trendline Annotations Module
 * Handles volume annotation creation and display
 */

import { state } from '../core/state.js';
import { formatVolume } from './geometry.js';

/**
 * Create annotation for a trendline showing volume
 */
export function createAnnotation(line, chartData) {
  const midX = new Date((new Date(line.x0).getTime() + new Date(line.x1).getTime()) / 2);
  const midY = (line.y0 + line.y1) / 2;
  const priceRange = Math.abs(line.y1 - line.y0);
  const baseOffset = Math.max(priceRange * 0.08, 5);

  let yPosition = midY + baseOffset;
  let verticalAlign = 'bottom';

  if (chartData && chartData.length > 0) {
    const nearbyCandlesAbove = chartData.filter(candle => {
      return candle.High > midY && candle.High < (midY + baseOffset * 2);
    });

    const nearbyCandlesBelow = chartData.filter(candle => {
      return candle.Low < midY && candle.Low > (midY - baseOffset * 2);
    });

    if (nearbyCandlesAbove.length > nearbyCandlesBelow.length) {
      yPosition = midY - baseOffset;
      verticalAlign = 'top';
    }
  }

  return {
    x: midX,
    y: yPosition,
    text: formatVolume(line.volume),
    showarrow: false,
    font: {
      color: state.currentTheme === 'light' ? "#2563eb" : "#60a5fa",
      size: 14,
      family: 'Arial, sans-serif',
      weight: 'bold'
    },
    xanchor: 'center',
    yanchor: verticalAlign,
    bgcolor: state.currentTheme === 'light' ? 'rgba(255, 255, 255, 0.95)' : 'rgba(30, 41, 59, 0.95)',
    borderpad: 6,
    bordercolor: state.currentTheme === 'light' ? '#2563eb' : '#60a5fa',
    borderwidth: 2,
    captureevents: true,
    clicktoshow: false
  };
}

/**
 * Setup click handlers for annotations (DOM-based)
 */
export function setupAnnotationClicks(plotId = "plot") {
  setTimeout(() => {
    // console.log('Setting up annotation clicks...');
    const plotDiv = document.getElementById(plotId);

    if (!plotDiv) {
      console.warn(`Plot element with id "${plotId}" not found`);
      return;
    }

    // Get all SVG text elements
    const allTextElements = document.querySelectorAll(`#${plotId} svg text`);
    // console.log(`Found ${allTextElements.length} SVG text elements`);

    // Get our volume annotations from layout
    const layoutAnns = (plotDiv.layout && plotDiv.layout.annotations) || [];
    const volumeTexts = layoutAnns
      .filter(a => a._lineId)
      .map(a => a.text);

    // console.log('Volume texts to look for:', volumeTexts);

    let foundCount = 0;
    allTextElements.forEach(textEl => {
      const text = textEl.textContent || textEl.innerText;

      // Check if this text matches one of our volume labels
      if (volumeTexts.includes(text)) {
        foundCount++;
        textEl.style.cursor = 'pointer';
        textEl.style.pointerEvents = 'all';

        // Remove existing listeners to avoid duplicates
        const newEl = textEl.cloneNode(true);
        textEl.parentNode.replaceChild(newEl, textEl);

        newEl.addEventListener('click', function(e) {
          e.stopPropagation();
          e.preventDefault();
          // console.log('✓ Volume label clicked:', text);

          // Find the line ID from the annotation text
          const layoutAnns = plotDiv.layout.annotations || [];
          for (let a of layoutAnns) {
            if (a._lineId && a.text === text) {
              state.selectedLineId = a._lineId;
              // Import dynamically to avoid circular dependency
              import('./selection.js').then(module => {
                module.updateShapesColors(plotId);
              });
              // console.log('✓ Selected line via volume label:', state.selectedLineId);
              return;
            }
          }
        });

        // console.log(`✓ Added click handler to volume label: "${text}"`);
      }
    });

    // console.log(`Attached ${foundCount} annotation click handlers`);
  }, 500);
}
