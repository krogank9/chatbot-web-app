// Chat state management
let currentChatId = localStorage.getItem('lastChatId');
let personas = [];
let currentResponse = null;
let chatToDelete = null;
let mediaRecorder = null;
let isRecording = false;
let silenceTimeout = null;
let audioChunks = [];
const SILENCE_THRESHOLD = 2000; // 2 seconds of silence before stopping

// Voice Chat Mode
let isVoiceModeActive = false;
let voiceVisualizationBars = [];
const NUM_BARS = 32;
const MIN_WORDS = 3;
let lastSpokenTime = 0;
const SILENCE_BEFORE_SEND = 1500; // 1.5 seconds of relative silence before sending
const NOISE_THRESHOLD = 15; // Increased from 5 to be more tolerant of background noise
let waitingForBot = false;
let canStartRecording = true;
let hasSpokenSomething = false; // Track if user has actually spoken
let silenceStartTime = null;
let isProcessingVoice = false;

// UI Elements
const newChatModal = new bootstrap.Modal(document.getElementById('newChatModal'));
const deleteChatModal = new bootstrap.Modal(document.getElementById('deleteChatModal'));

// Audio Queue Class
class AudioQueue {
    constructor() {
        this.queue = [];
        this.isPlaying = false;
        this.onQueueEmpty = null;
        this.onAudioStart = null;
    }

    add(audioElement) {
        this.queue.push(audioElement);
        if (!this.isPlaying) {
            this.playNext();
        }
    }

    playNext() {
        if (this.queue.length === 0) {
            this.isPlaying = false;
            if (this.onQueueEmpty) {
                this.onQueueEmpty();
            }
            return;
        }

        this.isPlaying = true;
        const audio = this.queue[0];

        audio.onplay = () => {
            if (this.onAudioStart) {
                this.onAudioStart();
            }
        };

        audio.onended = () => {
            setTimeout(() => {
                this.queue.shift();
                this.playNext();
            }, 100);
        };

        audio.play().catch(e => {
            console.error('Audio playback failed:', e);
            this.queue.shift();
            this.playNext();
        });
    }

    clear() {
        this.queue = [];
        this.isPlaying = false;
    }
}

// UI Utility Functions
function disableInputs() {
    const input = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');
    input.value = '';
    sendButton.disabled = true;
}

function enableInputs() {
    const input = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');
    sendButton.disabled = false;
}

function showTypingIndicator() {
    const container = document.getElementById('chatContainer');
    const typingIndicator = document.createElement('div');
    typingIndicator.className = 'message bot-message';
    typingIndicator.id = 'current-response';
    typingIndicator.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
    container.appendChild(typingIndicator);
    container.scrollTop = container.scrollHeight;
}

function updateCurrentPersona(personaId) {
    const persona = personas.find(p => p.id === personaId);
    if (persona) {
        document.getElementById('currentPersona').textContent = persona.name;
    }
}

function updateChatListSelection(chatId) {
    document.querySelectorAll('.chat-item').forEach(item => {
        item.classList.toggle('active', item.querySelector('.flex-grow-1').getAttribute('onclick').includes(chatId));
    });
}

function addMessage(message, role, audioUrl = null) {
    const container = document.getElementById('chatContainer');
    const div = document.createElement('div');
    div.className = `message ${role === 'user' ? 'user-message' : 'bot-message'}`;
    div.style.whiteSpace = 'pre-wrap';
    div.textContent = message;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
    return div;
}

// Chat API Functions
async function handleChatResponse(message, audioQueue, state) {
    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                message: message,
                chat_id: currentChatId
            })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8', { stream: true });
        let buffer = '';

        while (true) {
            const {done, value} = await reader.read();
            if (done || state.aborted) break;

            // Decode the chunk and add it to our buffer
            buffer += decoder.decode(value, { stream: true });
            
            // Split on double newlines (SSE standard) and process each complete event
            const events = buffer.split('\n\n');
            buffer = events.pop(); // Keep the last partial event in the buffer
            
            for (const event of events) {
                const line = event;
                if (line.startsWith('data: ')) {
                    const data = JSON.parse(line.slice(6));
                    
                    if (!data.done) {
                        if (!state.hasStartedResponding) {
                            // For the first chunk, remove any leading whitespace
                            state.fullResponse = data.text;
                            document.getElementById('current-response')?.remove();
                            state.currentMessage = addMessage(state.fullResponse, 'assistant');
                            state.hasStartedResponding = true;
                        } else {
                            // For subsequent chunks, preserve newlines but handle concatenation carefully
                            state.fullResponse += data.text;
                            if (state.currentMessage) {
                                state.currentMessage.textContent = state.fullResponse;
                            }
                        }

                        if (data.audio) {
                            const audioElement = document.createElement('audio');
                            audioElement.src = data.audio.audio_path;
                            audioQueue.add(audioElement);
                        }
                    }
                }
            }
        }

        // Process any remaining data
        const remaining = decoder.decode();
        if (remaining) {
            const line = remaining;
            if (line.startsWith('data: ')) {
                const data = JSON.parse(line.slice(6));
                if (!data.done && data.text) {
                    state.fullResponse += data.text;
                    if (state.currentMessage) {
                        state.currentMessage.textContent = state.fullResponse;
                    }
                }
            }
        }

        if (state.fullResponse) {
            await fetch('/generate_audio', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: state.fullResponse })
            });
        }
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('current-response')?.remove();
    }
}

// Chat Management Functions
function showDeleteChatModal(chatId) {
    chatToDelete = chatId;
    deleteChatModal.show();
}

function confirmDeleteChat() {
    if (!chatToDelete) return;

    fetch(`/chats/${chatToDelete}`, {
        method: 'DELETE'
    })
    .then(response => {
        if (response.ok) {
            if (chatToDelete === currentChatId) {
                currentChatId = null;
                localStorage.removeItem('lastChatId');
                document.getElementById('chatContainer').innerHTML = '';
                document.getElementById('currentPersona').textContent = 'Select a chat to begin';
                document.getElementById('userInput').disabled = true;
                document.getElementById('sendButton').disabled = true;
            }
            loadChats();
        } else {
            alert('Failed to delete chat');
        }
        deleteChatModal.hide();
        chatToDelete = null;
    });
}

// Chat UI Functions
function toggleSidebar() {
    document.getElementById('sidebar').classList.toggle('show');
}

function loadPersonas() {
    fetch('/personas')
        .then(response => response.json())
        .then(data => {
            personas = data;
            const select = document.getElementById('personaSelect');
            select.innerHTML = personas.map(p => 
                `<option value="${p.id}">${p.name}</option>`
            ).join('');
            updatePersonaDescription();
        });
}

function updatePersonaDescription() {
    const personaId = document.getElementById('personaSelect').value;
    const persona = personas.find(p => p.id === personaId);
    if (persona) {
        document.getElementById('personaDescription').textContent = persona.description;
    }
}

function showNewChatModal() {
    loadPersonas();
    newChatModal.show();
}

function createNewChat() {
    const personaId = document.getElementById('personaSelect').value;
    fetch('/chats', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ persona_id: personaId })
    })
    .then(response => response.json())
    .then(chat => {
        newChatModal.hide();
        loadChats();
        switchToChat(chat.id);
    });
}

// Chat Management Functions
function loadChats() {
    fetch('/chats')
        .then(response => response.json())
        .then(chats => {
            const chatList = document.getElementById('chatList');
            chatList.innerHTML = chats.map(chat => `
                <div class="chat-item ${chat.id === currentChatId ? 'active' : ''} d-flex align-items-start">
                    <div class="flex-grow-1" onclick="switchToChat('${chat.id}')">
                        <div class="fw-bold">${chat.persona_name}</div>
                        <small class="text-muted">${new Date(chat.updated_at).toLocaleString()}</small>
                    </div>
                    <button class="btn btn-sm btn-outline-danger ms-2" 
                            onclick="showDeleteChatModal('${chat.id}')"
                            style="padding: 0.25rem 0.5rem;">
                        <i class="bi bi-trash"></i>
                    </button>
                </div>
            `).join('');
            
            if (!currentChatId && chats.length > 0) {
                switchToChat(chats[0].id);
            } else if (currentChatId && !document.getElementById('chatContainer').children.length) {
                const currentChat = chats.find(c => c.id === currentChatId);
                if (currentChat) {
                    switchToChat(currentChatId, false);
                } else if (chats.length > 0) {
                    switchToChat(chats[0].id);
                }
            }
        });
}

function switchToChat(chatId, saveToStorage = true) {
    if (currentResponse) {
        currentResponse();
        currentResponse = null;
    }

    currentChatId = chatId;
    if (saveToStorage) {
        localStorage.setItem('lastChatId', chatId);
    }
    
    const userInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');
    userInput.disabled = false;
    sendButton.disabled = false;
    
    fetch(`/chats/${chatId}`)
        .then(response => response.json())
        .then(chat => {
            updateChatUI(chat);
        });
}

function updateChatUI(chat) {
    const container = document.getElementById('chatContainer');
    container.innerHTML = chat.messages.map(msg => 
        createMessageElement(msg)
    ).join('');
    container.scrollTop = container.scrollHeight;
    
    updateCurrentPersona(chat.persona_id);
    updateChatListSelection(chat.id);
}

function createMessageElement(msg) {
    return `<div class="message ${msg.role === 'user' ? 'user-message' : 'bot-message'}" style="white-space: pre-wrap">${msg.content.trim()}</div>`;
}

// Add new function for resetting voice states
function resetVoiceStates() {
    isRecording = false;
    isProcessingVoice = false;
    waitingForBot = false;
    canStartRecording = true;
    hasSpokenSomething = false;
    silenceStartTime = null;
    audioChunks = [];
    
    // Clear any pending timeouts
    if (silenceTimeout) {
        clearTimeout(silenceTimeout);
        silenceTimeout = null;
    }
    
    // Reset UI elements if in voice mode
    if (isVoiceModeActive) {
        const visualization = document.getElementById('voiceVisualization');
        visualization.classList.remove('listening', 'bot-speaking');
        document.getElementById('voiceText').textContent = '';
    }
}

function sendMessage() {
    const input = document.getElementById('userInput');
    const message = input.value.trim();
    if (!message || !currentChatId || document.getElementById('current-response')) {
        return;
    }

    disableInputs();
    addMessage(message, 'user');
    showTypingIndicator();

    const audioQueue = new AudioQueue();
    let fullResponse = '';
    let currentMessage = null;
    let audioContainer = null;
    let hasStartedResponding = false;
    let aborted = false;

    if (isVoiceModeActive) {
        // Immediately set waiting states when bot starts responding
        waitingForBot = true;
        canStartRecording = false;
        isProcessingVoice = false;  // Reset this as we're starting a new response
        hasSpokenSomething = false; // Reset this for next turn
        
        // Stop any ongoing recording
        if (isRecording) {
            stopRecording();
        }

        // Update UI to show bot is responding
        document.getElementById('voiceStatus').textContent = 'Thinking...';
        document.getElementById('voiceText').textContent = '';
        document.getElementById('voiceVisualization').classList.add('bot-speaking');
        document.getElementById('voiceVisualization').classList.remove('listening');

        // Handle audio queue events
        audioQueue.onQueueEmpty = () => {
            if (isVoiceModeActive) {
                // Add a delay before starting next recording
                setTimeout(() => {
                    if (isVoiceModeActive) {
                        // Only reset states and start recording if we're still in voice mode
                        resetVoiceStates();
                        document.getElementById('voiceStatus').textContent = 'Listening...';
                        document.getElementById('voiceVisualization').classList.remove('bot-speaking');
                        waitingForBot = false;
                        canStartRecording = true;
                        toggleRecording();
                    }
                }, 1500);
            }
        };

        audioQueue.onAudioStart = () => {
            if (isVoiceModeActive) {
                // Ensure we're in the correct state when bot is speaking
                waitingForBot = true;
                canStartRecording = false;
                isProcessingVoice = false;
                if (isRecording) {
                    stopRecording();
                }
                document.getElementById('voiceStatus').textContent = 'Speaking...';
                document.getElementById('voiceVisualization').classList.add('bot-speaking');
                document.getElementById('voiceVisualization').classList.remove('listening');
            }
        };
    }

    // Create a promise to track when the entire response is complete
    let responseComplete = false;
    
    handleChatResponse(message, audioQueue, {
        fullResponse,
        currentMessage,
        audioContainer,
        hasStartedResponding,
        aborted
    }).then(() => {
        responseComplete = true;
        if (!isVoiceModeActive) {
            enableInputs();
        }
        input.value = '';
    });
}

function toggleRecording() {
    // If we're already recording, waiting for bot, or bot is speaking, don't do anything
    if (isRecording || waitingForBot || !canStartRecording) {
        return;
    }
    
    // If voice mode isn't active, enter it
    if (!isVoiceModeActive) {
        enterVoiceMode();
    }
    
    // Only start recording if we're not waiting for bot and not processing
    if (!waitingForBot && !isProcessingVoice && canStartRecording) {
        // Reset state
        resetVoiceStates();
        
        try {
            mediaRecorder.start(100);
            isRecording = true;
            
            const voiceStatus = document.getElementById('voiceStatus');
            voiceStatus.textContent = 'Listening...';
            document.getElementById('voiceVisualization').classList.add('listening');
            document.getElementById('voiceVisualization').classList.remove('bot-speaking');
            document.getElementById('voiceText').textContent = '';
            
            window.checkSilence();
        } catch (error) {
            console.error('Failed to start recording:', error);
            document.getElementById('voiceStatus').textContent = 'Failed to start recording';
            exitVoiceMode();
        }
    }
}

function exitVoiceMode() {
    isVoiceModeActive = false;
    
    // Stop any ongoing recording and clean up
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
    }
    if (mediaRecorder) {
        const tracks = mediaRecorder.stream.getTracks();
        tracks.forEach(track => track.stop());
        mediaRecorder = null;
    }

    // Reset all states
    resetVoiceStates();
    
    // Reset UI elements
    document.getElementById('voiceChatMode').classList.remove('active');
    const visualization = document.getElementById('voiceVisualization');
    visualization.classList.remove('listening', 'bot-speaking');
    document.getElementById('voiceStatus').textContent = '';
    document.getElementById('voiceText').textContent = '';
    
    // Re-enable text input
    const input = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');
    input.disabled = false;
    sendButton.disabled = false;
}

// Voice Recording Functions
function setupVisualizationBars() {
    const container = document.getElementById('visualizationBars');
    container.innerHTML = '';
    voiceVisualizationBars = [];
    
    for (let i = 0; i < NUM_BARS; i++) {
        const bar = document.createElement('div');
        bar.className = 'visualization-bar';
        container.appendChild(bar);
        voiceVisualizationBars.push(bar);
    }
}

function updateVisualization(dataArray) {
    const step = Math.floor(dataArray.length / NUM_BARS);
    
    for (let i = 0; i < NUM_BARS; i++) {
        const start = i * step;
        const end = start + step;
        let sum = 0;
        
        for (let j = start; j < end; j++) {
            sum += dataArray[j];
        }
        
        const average = sum / step;
        const height = Math.max(3, (average / 255) * 60);
        voiceVisualizationBars[i].style.height = `${height}px`;
    }
}

function enterVoiceMode() {
    isVoiceModeActive = true;
    document.getElementById('voiceChatMode').classList.add('active');
    setupVisualizationBars();
}

async function setupVoiceRecording() {
    try {
        // First check if STT model is loaded
        const response = await fetch('/models/loaded');
        const loadedModels = await response.json();
        
        if (!loadedModels.stt) {
            console.error('Speech-to-text model not loaded');
            const micButton = document.getElementById('micButton');
            micButton.title = 'Speech-to-text model not loaded';
            return;
        }

        // Remove any existing click handlers and add a new one
        const micButton = document.getElementById('micButton');
        micButton.disabled = false;
        micButton.title = 'Click to start recording';
        const newMicButton = micButton.cloneNode(true);
        micButton.parentNode.replaceChild(newMicButton, micButton);
        newMicButton.addEventListener('click', async () => {
            // If we're not in voice mode, set up new recording
            if (!isVoiceModeActive) {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ 
                        audio: {
                            channelCount: 1,
                            sampleRate: 16000,
                            sampleSize: 16,
                            echoCancellation: true,
                            noiseSuppression: true
                        } 
                    });
                    
                    mediaRecorder = new MediaRecorder(stream, {
                        mimeType: 'audio/webm;codecs=opus'
                    });
                    
                    mediaRecorder.ondataavailable = (event) => {
                        if (event.data.size > 0) {
                            audioChunks.push(event.data);
                        }
                    };

                    mediaRecorder.onstop = async () => {
                        // Only process if we've actually spoken or are processing voice
                        if (hasSpokenSomething && isProcessingVoice) {
                            const voiceStatus = document.getElementById('voiceStatus');
                            voiceStatus.textContent = 'Thinking...';
                            
                            try {
                                const audioBlob = new Blob(audioChunks, { type: 'audio/webm;codecs=opus' });
                                const audioContext = new AudioContext();
                                const audioData = await audioBlob.arrayBuffer();
                                const audioBuffer = await audioContext.decodeAudioData(audioData);
                                
                                const wavBlob = await convertToWav(audioBuffer);
                                const formData = new FormData();
                                formData.append('audio', wavBlob, 'recording.wav');

                                const response = await fetch('/transcribe', {
                                    method: 'POST',
                                    body: formData
                                });
                                
                                const data = await response.json();
                                if (data.text) {
                                    const words = data.text.trim().split(/\s+/);
                                    if (words.length >= MIN_WORDS) {
                                        voiceStatus.textContent = 'Thinking...';
                                        document.getElementById('voiceText').textContent = '';
                                        
                                        const input = document.getElementById('userInput');
                                        input.value = data.text;
                                        sendMessage();
                                    } else {
                                        voiceStatus.textContent = 'Message too short...';
                                        document.getElementById('voiceVisualization').classList.remove('listening');
                                        // Automatically restart recording after a short delay
                                        setTimeout(() => {
                                            if (isVoiceModeActive) {
                                                resetVoiceStates();
                                                toggleRecording();
                                            }
                                        }, 1500);
                                    }
                                } else if (data.error) {
                                    console.error('Transcription error:', data.error);
                                    voiceStatus.textContent = 'Failed to understand...';
                                    setTimeout(() => {
                                        if (isVoiceModeActive) {
                                            resetVoiceStates();
                                            toggleRecording();
                                        }
                                    }, 1500);
                                }
                            } catch (error) {
                                console.error('Transcription error:', error);
                                voiceStatus.textContent = 'Error occurred...';
                                setTimeout(() => {
                                    if (isVoiceModeActive) {
                                        resetVoiceStates();
                                        toggleRecording();
                                    }
                                }, 1500);
                            }
                        }
                        // Always clean up audio chunks
                        audioChunks = [];
                    };

                    // Setup audio analysis for silence detection
                    const audioContext = new AudioContext();
                    const audioSource = audioContext.createMediaStreamSource(stream);
                    const analyser = audioContext.createAnalyser();
                    analyser.fftSize = 2048;
                    audioSource.connect(analyser);

                    window.checkSilence = function() {
                        // Don't process if we're not in a valid state for recording
                        if (!isRecording || waitingForBot || !canStartRecording || isProcessingVoice) {
                            return;
                        }

                        const dataArray = new Uint8Array(analyser.frequencyBinCount);
                        analyser.getByteFrequencyData(dataArray);
                        
                        // Update visualization only if we're actively listening
                        if (isVoiceModeActive && !waitingForBot) {
                            updateVisualization(dataArray);
                        }
                        
                        // Calculate average volume
                        const avgVolume = dataArray.reduce((a, b) => a + b, 0) / dataArray.length;
                        
                        // Detect if user is speaking
                        if (avgVolume > NOISE_THRESHOLD) {
                            hasSpokenSomething = true;
                            silenceStartTime = null;
                            if (silenceTimeout) {
                                clearTimeout(silenceTimeout);
                                silenceTimeout = null;
                            }
                        } else if (hasSpokenSomething && !silenceStartTime && !isProcessingVoice) {
                            // Start tracking silence only if we've spoken and aren't processing
                            silenceStartTime = Date.now();
                        }

                        // Check if we've been silent long enough to send
                        if (hasSpokenSomething && silenceStartTime && !isProcessingVoice) {
                            const silenceDuration = Date.now() - silenceStartTime;
                            if (silenceDuration >= SILENCE_BEFORE_SEND) {
                                isProcessingVoice = true;
                                stopRecording();
                                return;
                            }
                        }

                        // Only continue checking if we're in a valid state
                        if (isRecording && !waitingForBot && canStartRecording && !isProcessingVoice) {
                            requestAnimationFrame(checkSilence);
                        }
                    };
                } catch (error) {
                    console.error('Error setting up recording:', error);
                    return;
                }
            }
            toggleRecording();
        });

        // Setup exit button
        document.getElementById('exitVoiceChat').addEventListener('click', exitVoiceMode);

    } catch (error) {
        console.error('Error setting up voice recording:', error);
        const micButton = document.getElementById('micButton');
        micButton.title = 'Error: ' + error.message;
        micButton.disabled = true;
    }
}

// Helper function to convert AudioBuffer to WAV format
function convertToWav(audioBuffer) {
    const numOfChannels = 1;  // Force mono
    const sampleRate = 16000;  // Force 16kHz
    const bitsPerSample = 16;
    const bytesPerSample = bitsPerSample / 8;
    const length = audioBuffer.length * bytesPerSample;
    const buffer = new ArrayBuffer(44 + length);
    const view = new DataView(buffer);
    
    // Write WAV header
    writeUTFBytes(view, 0, 'RIFF');                     // RIFF identifier
    view.setUint32(4, 36 + length, true);              // File length
    writeUTFBytes(view, 8, 'WAVE');                     // WAVE identifier
    writeUTFBytes(view, 12, 'fmt ');                    // fmt chunk
    view.setUint32(16, 16, true);                      // Length of format chunk
    view.setUint16(20, 1, true);                       // Format type (1 = PCM)
    view.setUint16(22, numOfChannels, true);           // Number of channels
    view.setUint32(24, sampleRate, true);              // Sample rate
    view.setUint32(28, sampleRate * bytesPerSample, true); // Byte rate
    view.setUint16(32, numOfChannels * bytesPerSample, true); // Block align
    view.setUint16(34, bitsPerSample, true);           // Bits per sample
    writeUTFBytes(view, 36, 'data');                   // data chunk identifier
    view.setUint32(40, length, true);                  // data chunk length
    
    // Get audio data and convert to mono if needed
    let audioData;
    if (audioBuffer.numberOfChannels > 1) {
        // Mix down to mono
        const channel1 = audioBuffer.getChannelData(0);
        const channel2 = audioBuffer.getChannelData(1);
        audioData = new Float32Array(audioBuffer.length);
        for (let i = 0; i < audioBuffer.length; i++) {
            audioData[i] = (channel1[i] + channel2[i]) / 2;
        }
    } else {
        audioData = audioBuffer.getChannelData(0);
    }
    
    // Resample to 16kHz if needed
    if (audioBuffer.sampleRate !== sampleRate) {
        audioData = resampleAudio(audioData, audioBuffer.sampleRate, sampleRate);
    }
    
    // Convert to 16-bit PCM
    let offset = 44;
    for (let i = 0; i < audioData.length; i++) {
        const sample = Math.max(-1, Math.min(1, audioData[i]));
        view.setInt16(offset, sample * 0x7FFF, true);
        offset += 2;
    }
    
    return new Blob([buffer], { type: 'audio/wav' });
}

// Helper function for resampling audio
function resampleAudio(audioData, fromSampleRate, toSampleRate) {
    const ratio = fromSampleRate / toSampleRate;
    const newLength = Math.round(audioData.length / ratio);
    const result = new Float32Array(newLength);
    
    for (let i = 0; i < newLength; i++) {
        const position = i * ratio;
        const index = Math.floor(position);
        const fraction = position - index;
        
        if (index + 1 < audioData.length) {
            result[i] = audioData[index] * (1 - fraction) + audioData[index + 1] * fraction;
        } else {
            result[i] = audioData[index];
        }
    }
    
    return result;
}

function writeUTFBytes(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}

function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;
        document.getElementById('voiceVisualization').classList.remove('listening');
        
        if (silenceTimeout) {
            clearTimeout(silenceTimeout);
            silenceTimeout = null;
        }
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadChats();
    loadPersonas();
    setupVoiceRecording();
    document.getElementById('personaSelect').addEventListener('change', updatePersonaDescription);
}); 