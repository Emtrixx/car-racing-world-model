
document.addEventListener('DOMContentLoaded', () => {
    const modelSelect = document.getElementById('model-select');
    const startBtn = document.getElementById('start-btn');
    const canvas = document.getElementById('dream-canvas');
    const ctx = canvas.getContext('2d');

    let dreamInterval = null;

    startBtn.addEventListener('click', async () => {
        if (dreamInterval) {
            clearInterval(dreamInterval);
        }

        const modelType = modelSelect.value;
        const response = await fetch(`/api/v1/dream/start/${modelType}`, { method: 'POST' });
        const data = await response.json();
        image.src = `data:image/jpeg;base64,${data.frame}`;
        
        dreamInterval = setInterval(gameLoop, 1000 / 8); // 8 FPS
    });

    document.addEventListener('keydown', (e) => {
        keysPressed[e.key] = true;
    });

    document.addEventListener('keyup', (e) => {
        keysPressed[e.key] = false;
    });

    async function gameLoop() {
        const steer = (keysPressed['ArrowLeft'] ? -1.0 : 0.0) + (keysPressed['ArrowRight'] ? 1.0 : 0.0);
        const gas = keysPressed['ArrowUp'] ? 0.8 : -1.0;
        const brake = keysPressed['ArrowDown'] ? 0.2 : -1.0;

        const action = [steer, gas, brake];

        const response = await fetch('/api/v1/dream/step', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ action })
        });

        if (response.ok) {
            const data = await response.json();
            image.src = `data:image/jpeg;base64,${data.frame}`;
        } else {
            clearInterval(dreamInterval);
            console.error("Error during dream step");
        }
    }
});
