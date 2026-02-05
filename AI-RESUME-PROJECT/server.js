require('dotenv').config();
const express = require('express');
const cors = require('cors');
const app = express();
const PORT = 3000;

app.use(cors());
// Resume bada ho sakta hai, isliye limit 10MB 
app.use(express.json({ limit: '10mb' })); 

app.post('/api/generate', async (req, res) => {
    try {
        const { prompt } = req.body;
        const apiKey = process.env.GEMINI_API_KEY;
        
       //API MODEL KA NAAM
        const modelName = "gemini-2.5-pro"; 
        
        const url = `https://generativelanguage.googleapis.com/v1beta/models/${modelName}:generateContent?key=${apiKey}`;

        const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                contents: [{ parts: [{ text: prompt }] }]
            })
        });

        const data = await response.json();

        if (data.error) {
            // Agar Google ne mana kiya, toh uska reason bhejo
            console.error("Gemini API Error:", data.error.message);
            return res.status(400).json({ error: data.error.message });
        }

        if (data.candidates && data.candidates.length > 0) {
            const aiText = data.candidates[0].content.parts[0].text;
            res.json({ result: aiText });
        } else {
            res.status(500).json({ error: "AI ne koi jawab nahi diya (Empty Response)." });
        }

    } catch (error) {
        console.error("Server Error:", error.message);
        res.status(500).json({ error: error.message });
    }
});

app.listen(PORT, () => {
    console.log(`✅ Server (Pro Model) running at http://localhost:${PORT}`);
});