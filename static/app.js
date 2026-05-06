/**
 * Token sequence renderer: colors tokens by KL intensity
 */
function renderTokenSequence(el) {
    const tokens = JSON.parse(el.dataset.tokens);
    const kl = JSON.parse(el.dataset.kl);
    const klMax = parseFloat(el.dataset.klMax);
    const position = parseInt(el.dataset.position);
    const bandColor = el.dataset.bandColor;

    el.innerHTML = '';

    tokens.forEach((tok, i) => {
        const span = document.createElement('span');
        span.className = 'tok';
        span.textContent = tok;

        // KL-intensity green coloring: intensity = min(1, sqrt(kl/klMax))
        if (klMax > 0 && i < kl.length) {
            const intensity = Math.min(1, Math.sqrt(kl[i] / klMax));
            if (intensity > 0.01) {
                span.style.backgroundColor = `rgba(76, 191, 102, ${intensity * 0.8})`;
                // Use dark text when background is bright
                if (intensity > 0.5) {
                    span.style.color = '#1a1b26';
                }
            }
        }

        // Highlight target position with band-color dashed border
        if (i === position) {
            span.className += ' tok-target';
            span.style.outlineColor = bandColor;
        }

        el.appendChild(span);
    });
}
