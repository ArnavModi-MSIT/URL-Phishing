document.getElementById('urlForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const urlInput = document.getElementById('urlInput');
    const resultDiv = document.getElementById('result');
    const resultText = document.getElementById('resultText');

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
    } catch (error) {
        resultDiv.classList.remove('hidden');
        resultText.textContent = 'Error checking URL';
        resultText.style.color = 'orange';
    }
});