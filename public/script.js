const API_URL = "postgresql://phishing_feedback_db_user:YU0q5xSMwbvrMvgnMvZpjHnb4LRUGxAO@dpg-cundop23esus73cg5up0-a/phishing_feedback_db"; // Replace with your actual Render URL

document.getElementById('urlForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const urlInput = document.getElementById('urlInput');
    const resultDiv = document.getElementById('result');
    const resultText = document.getElementById('resultText');
    const feedbackDiv = document.getElementById('feedback');
    const feedbackYes = document.getElementById('feedbackYes');
    const feedbackNo = document.getElementById('feedbackNo');

    try {
        const response = await fetch(`${API_URL}/predict`, {  // 🔹 Use full URL
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
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
            await fetch(`postgresql://phishing_feedback_db_user:YU0q5xSMwbvrMvgnMvZpjHnb4LRUGxAO@dpg-cundop23esus73cg5up0-a/phishing_feedback_db/feedback`, {  // 🔹 Use full URL
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    url: urlInput.value,
                    is_phishing: data.is_phishing, 
                    feedback: true
                })
            });
            feedbackDiv.classList.add('hidden');
        };
        
        feedbackNo.onclick = async () => {
            await fetch(`postgresql://phishing_feedback_db_user:YU0q5xSMwbvrMvgnMvZpjHnb4LRUGxAO@dpg-cundop23esus73cg5up0-a/phishing_feedback_db/feedback`, {  // 🔹 Use full URL
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    url: urlInput.value,
                    is_phishing: data.is_phishing, 
                    feedback: false
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
