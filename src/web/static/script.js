document.addEventListener('DOMContentLoaded', () => {
    const modelSelect = document.getElementById('model-select');
    const startBtn = document.getElementById('start-btn');
    const resetBtn = document.getElementById('reset-btn');
    const canvas = document.getElementById('dream-canvas');
    const ctx = canvas.getContext('2d');
    const loader = document.getElementById('loader');
    const statusMessage = document.getElementById('status-message');
    const controlButtons = document.querySelectorAll('.control-btn');

    let keysPressed = {};
    const image = new Image();
    let ws = null;
    let gameLoopId = null;

    let lastFrameTime = 0;
    const frameInterval = 1000 / 8; // 8 FPS

    // This is the most reliable place to handle UI changes after a frame is ready.
    image.onload = () => {
        ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

        // If loader is visible, hide it. This is the core fix.
        if (!loader.classList.contains('hidden')) {
            showLoader(false);
            setControlsEnabled(true);
            setStatus('Dream started! Use arrow keys to drive.');
        }
    };

    function showLoader(show) {
        loader.classList.toggle('hidden', !show);
        // loader.hidden = !show;
    }

    function setControlsEnabled(isRunning) {
        startBtn.disabled = isRunning;
        // modelSelect is always enabled
        resetBtn.disabled = !isRunning;
    }

    function setStatus(message, isError = false) {
        statusMessage.textContent = message;
        statusMessage.style.color = isError ? '#d93025' : '#606770';
    }

    function stopGameLoop() {
        if (gameLoopId) {
            cancelAnimationFrame(gameLoopId);
            gameLoopId = null;
        }
    }

    function startGameLoop() {
        stopGameLoop(); // Ensure no multiple loops are running
        lastFrameTime = performance.now();

        function loop(currentTime) {
            gameLoopId = requestAnimationFrame(loop);

            if (!ws || ws.readyState !== WebSocket.OPEN) {
                return;
            }

            const elapsed = currentTime - lastFrameTime;
            if (elapsed > frameInterval) {
                lastFrameTime = currentTime - (elapsed % frameInterval);

                let steer = 0.0;
                let gas = -1.0;
                let brake = -1.0;

                if (keysPressed['ArrowUp']) {
                    gas = 0.8;
                }
                if (keysPressed['ArrowDown']) {
                    brake = 0.2;
                }
                if (keysPressed['ArrowLeft']) {
                    steer = -1.0;
                }
                if (keysPressed['ArrowRight']) {
                    steer = 1.0;
                }

                const action = [steer, gas, brake];

                ws.send(JSON.stringify({type: 'step', action: action}));
            }
        }

        gameLoopId = requestAnimationFrame(loop);
    }

    function connectWebSocket(sessionId) {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/${sessionId}`;
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            setStatus('Connection established. Starting dream...');
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.frame) {
                // This triggers the image.onload handler where the magic happens
                image.src = `data:image/jpeg;base64,${data.frame}`;
            }

            if (!gameLoopId) {
                startGameLoop();
            }
        };

        ws.onclose = () => {
            setStatus('Dream has ended. Please start a new one.', true);
            setControlsEnabled(false);
            stopGameLoop();
            ws = null;
        };

        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            setStatus('An error occurred with the connection.', true);
            setControlsEnabled(false);
            stopGameLoop();
        };
    }

    async function startNewDream() {
        if (ws) {
            ws.close();
        }
        stopGameLoop();
        setControlsEnabled(false);
        resetBtn.disabled = true; // Disable reset explicitly
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
            connectWebSocket(data.session_id);
        } catch (error) {
            console.error('Start error:', error);
            setStatus(`Error: ${error.message}`, true);
            setControlsEnabled(false);
            showLoader(false);
        }
    }

    startBtn.addEventListener('click', startNewDream);

    resetBtn.addEventListener('click', () => {
        if (ws) {
            stopGameLoop();
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

    // Explicitly hide loader on initial page load to prevent it from showing briefly.
    showLoader(false);

    modelSelect.addEventListener('change', () => {
        // If a game is currently running, changing the model triggers a restart with the new model.
        if (ws && ws.readyState === WebSocket.OPEN) {
            startNewDream();
        }
    });

    const controlMappings = {
        up: 'ArrowUp',
        down: 'ArrowDown',
        left: 'ArrowLeft',
        right: 'ArrowRight'
    };

    function setVirtualKeyState(button, isActive) {
        if (!button) {
            return;
        }
        const action = button.dataset.action;
        const key = controlMappings[action];
        if (!key) {
            return;
        }
        keysPressed[key] = isActive;
        button.classList.toggle('active', isActive);
    }

    function bindPointerControls() {
        controlButtons.forEach((button) => {
            button.addEventListener('pointerdown', (event) => {
                event.preventDefault();
                setVirtualKeyState(button, true);

                const endHandler = (endEvent) => {
                    if (endEvent.pointerId !== event.pointerId) {
                        return;
                    }
                    setVirtualKeyState(button, false);
                    window.removeEventListener('pointerup', endHandler);
                    window.removeEventListener('pointercancel', endHandler);
                };

                window.addEventListener('pointerup', endHandler);
                window.addEventListener('pointercancel', endHandler);
            });

            button.addEventListener('contextmenu', (event) => event.preventDefault());
        });
    }

    function bindFallbackControls() {
        controlButtons.forEach((button) => {
            button.addEventListener('touchstart', (event) => {
                event.preventDefault();
                setVirtualKeyState(button, true);
            }, {passive: false});
            button.addEventListener('touchend', () => setVirtualKeyState(button, false));
            button.addEventListener('touchcancel', () => setVirtualKeyState(button, false));
            button.addEventListener('mousedown', (event) => {
                event.preventDefault();
                setVirtualKeyState(button, true);
            });
            button.addEventListener('mouseup', () => setVirtualKeyState(button, false));
            button.addEventListener('mouseleave', () => setVirtualKeyState(button, false));
            button.addEventListener('contextmenu', (event) => event.preventDefault());
        });
    }

    if (window.PointerEvent) {
        bindPointerControls();
    } else if (controlButtons.length) {
        bindFallbackControls();
    }
});
