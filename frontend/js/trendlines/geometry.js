// Geometry Utilities for Trendlines

export function formatVolume(vol) {
  if (vol >= 1e9) return (vol / 1e9).toFixed(2) + 'B';
  if (vol >= 1e6) return (vol / 1e6).toFixed(2) + 'M';
  if (vol >= 1e3) return (vol / 1e3).toFixed(2) + 'K';
  return vol.toFixed(0);
}

export function getDataCoords(eventData, plotId = "plot") {
  const plotDiv = document.getElementById(plotId);

  if (!plotDiv || !plotDiv._fullLayout) {
    console.warn(`Plot element with id "${plotId}" not found or not initialized`);
    return null;
  }

  const xaxis = plotDiv._fullLayout.xaxis;
  const yaxis = plotDiv._fullLayout.yaxis;

  if (!xaxis || !yaxis) return null;

  const xRange = xaxis.range;
  const yRange = yaxis.range;
  const xDomain = xaxis.domain;
  const yDomain = yaxis.domain;

  const plotWidth = plotDiv.clientWidth;
  const plotHeight = plotDiv.clientHeight;

  const xPixelStart = xDomain[0] * plotWidth;
  const xPixelRange = (xDomain[1] - xDomain[0]) * plotWidth;
  const yPixelStart = (1 - yDomain[1]) * plotHeight;
  const yPixelRange = (yDomain[1] - yDomain[0]) * plotHeight;

  const xFraction = (eventData.x - xPixelStart) / xPixelRange;
  const yFraction = 1 - (eventData.y - yPixelStart) / yPixelRange;

  const xData = xRange[0] + xFraction * (xRange[1] - xRange[0]);
  const yData = yRange[0] + yFraction * (yRange[1] - yRange[0]);

  return { x: new Date(xData), y: yData };
}

export function distanceToSegment(px, py, x0, y0, x1, y1, xScale, yScale) {
  // Calculate line segment vector
  const dx = (x1 - x0) * xScale;
  const dy = (y1 - y0) * yScale;

  // Calculate squared length
  const lenSq = dx * dx + dy * dy;

  console.log('Distance calc:', { dx, dy, lenSq, xScale, yScale });

  // If line is a point
  if (lenSq === 0 || lenSq < 0.0001) {
    const distX = (px - x0) * xScale;
    const distY = (py - y0) * yScale;
    const dist = Math.sqrt(distX * distX + distY * distY);
    console.log('Point distance:', dist);
    return dist;
  }

  // Calculate projection parameter t
  const t = Math.max(0, Math.min(1,
    ((px - x0) * xScale * dx + (py - y0) * yScale * dy) / lenSq
  ));

  // Calculate closest point on segment
  const projX = x0 + t * (x1 - x0);
  const projY = y0 + t * (y1 - y0);

  // Calculate distance to closest point
  const distX = (px - projX) * xScale;
  const distY = (py - projY) * yScale;
  const dist = Math.sqrt(distX * distX + distY * distY);

  console.log('Segment distance:', dist, 't:', t);
  return dist;
}
