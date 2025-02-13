document.getElementById('urlForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const urlInput = document.getElementById('urlInput');
    const resultDiv = document.getElementById('result');
    const resultText = document.getElementById('resultText');
    const feedbackDiv = document.getElementById('feedback');
    const feedbackYes = document.getElementById('feedbackYes');
    const feedbackNo = document.getElementById('feedbackNo');

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ url: urlInput.value })
        });

        const data = await response.json();
        
        resultDiv.classList.remove('hidden');
        resultText.textContent = data.is_phishing 
            ? `Warning: High-Risk Phishing URL! (${data.risk_level})` 
            : `Safe URL (${data.risk_level})`;
        
        resultText.style.color = data.is_phishing ? 'red' : 'green';
        feedbackDiv.classList.remove('hidden');
        
        feedbackYes.onclick = async () => {
            await fetch('/feedback', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    url: urlInput.value,
                    is_phishing: data.is_phishing, 
                    feedback: true // ✅ Confirm the prediction
                })
            });
            feedbackDiv.classList.add('hidden');
        };
        
        feedbackNo.onclick = async () => {
            await fetch('/feedback', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    url: urlInput.value,
                    is_phishing: data.is_phishing, 
                    feedback: false // ✅ Opposite of the prediction
                })
            });
            feedbackDiv.classList.add('hidden');
        };
        
    } catch (error) {
        resultDiv.classList.remove('hidden');
        resultText.textContent = 'Error checking URL';
        resultText.style.color = 'orange';
    }
});
