document.addEventListener('DOMContentLoaded', () => {
    const modelSelect = document.getElementById('model-select');
    const startBtn = document.getElementById('start-btn');
    const resetBtn = document.getElementById('reset-btn');
    const canvas = document.getElementById('dream-canvas');
    const ctx = canvas.getContext('2d');
    const loader = document.getElementById('loader');
    const statusMessage = document.getElementById('status-message');

    let keysPressed = {};
    const image = new Image();
    let sessionId = null;
    let ws = null;

    let lastFrameTime = 0;
    const frameInterval = 1000 / 8; // 8 FPS

    image.onload = () => {
        ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    };

    function showLoader(show) {
        loader.classList.toggle('hidden', !show);
    }

    function setControlsEnabled(enabled) {
        startBtn.disabled = !enabled;
        modelSelect.disabled = !enabled;
    }

    function setStatus(message, isError = false) {
        statusMessage.textContent = message;
        statusMessage.style.color = isError ? '#d93025' : '#606770';
    }

    function connectWebSocket(sessionId) {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        ws = new WebSocket(`${protocol}//${window.location.host}/ws/${sessionId}`);

        ws.onopen = () => {
            setStatus('Connection established. Starting dream...');
            // The server will send the first frame automatically.
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.frame) {
                image.src = `data:image/jpeg;base64,${data.frame}`;
            }
            if (loader.style.display !== 'none') {
                showLoader(false);
                setStatus('Dream started! Use arrow keys to drive.');
            }
        };

        ws.onclose = () => {
            setStatus('Dream has ended. Please start a new one.', true);
            setControlsEnabled(true);
            resetBtn.disabled = true;
            ws = null;
        };

        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            setStatus('An error occurred with the connection.', true);
            setControlsEnabled(true);
            resetBtn.disabled = true;
        };
    }

    startBtn.addEventListener('click', async () => {
        if (ws) {
            ws.close();
        }
        setControlsEnabled(false);
        resetBtn.disabled = true;
        showLoader(true);
        setStatus('Initializing session...');

        const modelType = modelSelect.value;
        try {
            const response = await fetch(`/api/v1/dream/start/${modelType}`, {method: 'POST'});
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to start session');
            }
            const data = await response.json();
            sessionId = data.session_id;
            resetBtn.disabled = false;
            connectWebSocket(sessionId);
            lastFrameTime = performance.now();
            requestAnimationFrame(gameLoop);
        } catch (error) {
            console.error('Start error:', error);
            setStatus(`Error: ${error.message}`, true);
            setControlsEnabled(true);
            showLoader(false);
        }
    });

    resetBtn.addEventListener('click', () => {
        if (ws) {
            ws.send(JSON.stringify({type: 'reset'}));
            showLoader(true);
            setStatus('Resetting dream...');
        }
    });

    document.addEventListener('keydown', (e) => {
        if (e.key.startsWith('Arrow')) {
            e.preventDefault();
            keysPressed[e.key] = true;
        }
    });

    document.addEventListener('keyup', (e) => {
        if (e.key.startsWith('Arrow')) {
            e.preventDefault();
            keysPressed[e.key] = false;
        }
    });

    function gameLoop(currentTime) {
        if (!ws || ws.readyState !== WebSocket.OPEN) {
            return;
        }

        requestAnimationFrame(gameLoop);

        const elapsed = currentTime - lastFrameTime;
        if (elapsed > frameInterval) {
            lastFrameTime = currentTime - (elapsed % frameInterval);

            const steer = (keysPressed['ArrowLeft'] ? -1.0 : 0.0) + (keysPressed['ArrowRight'] ? 1.0 : 0.0);
            const gas = keysPressed['ArrowUp'] ? 1.0 : 0.0;
            const brake = keysPressed['ArrowDown'] ? 0.8 : 0.0;
            const action = [steer, gas, brake];

            ws.send(JSON.stringify({type: 'step', action: action}));
        }
    }
});
