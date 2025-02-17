function sendMessage() {
    const input = document.getElementById('user-input');
    const message = input.value.trim();
    
    if (message) {
        // Add user message to chat
        addMessage(message, 'user-message');
        
        // Clear input
        input.value = '';
        
        // Show typing indicator
        const typingIndicator = addMessage('Thinking...', 'bot-message');
        
        // Send message to server
        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message }),
        })
        .then(response => response.json())
        .then(data => {
            // Remove typing indicator
            typingIndicator.remove();
            
            // Add bot response to chat
            addMessage(data.response, 'bot-message');
            
            // Play audio response
            const audio = document.getElementById('response-audio');
            audio.src = data.audio_url + '?t=' + new Date().getTime();
        });
    }
}

function addMessage(text, className) {
    const messagesDiv = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message ' + className;
    messageDiv.textContent = text;
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
    return messageDiv;
}

// Allow sending message with Enter key
document.getElementById('user-input').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

// Add initial greeting message
document.addEventListener('DOMContentLoaded', function() {
    addMessage('Hello! How can I help you today?', 'bot-message');
});