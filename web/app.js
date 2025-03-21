const DB_THRESHOLD = -47;
let model = null;
let classNames = [];
let audioContext = null;
let mediaStream = null;
let appStartTime = new Date();
let startTime = null;
let quietTime = 0;
let noisyTime = 0;
let exposureData = new Map();
let db;

// Initialize IndexedDB
const initDb = () => {
    const request = indexedDB.open('SoundExposureDB', 1);

    request.onerror = (event) => {
        console.error('Database error:', event.target.error);
    };

    request.onupgradeneeded = (event) => {
        db = event.target.result;
        if (!db.objectStoreNames.contains('exposureStats')) {
            const store = db.createObjectStore('exposureStats', { keyPath: 'soundType' });
            store.createIndex('lastUpdate', 'lastUpdate', { unique: false });
        }
    };

    request.onsuccess = (event) => {
        db = event.target.result;
        loadHistoricalData(); // Load data when DB is ready
    };
};

// Save exposure data to IndexedDB
const saveExposureData = () => {
    if (!db) return;

    const transaction = db.transaction(['exposureStats'], 'readwrite');
    const store = transaction.objectStore('exposureStats');

    for (const [soundType, data] of exposureData.entries()) {
        store.put({
            soundType,
            totalDuration: data.totalDuration,
            avgDb: data.avgDb,
            maxDb: data.maxDb,
            count: data.count,
            lastUpdate: new Date().toISOString()
        });
    }
};

// Load historical data from IndexedDB
const loadHistoricalData = () => {
    if (!db) return;

    const transaction = db.transaction(['exposureStats'], 'readonly');
    const store = transaction.objectStore('exposureStats');
    const request = store.getAll();

    request.onsuccess = () => {
        const data = request.result;
        updateHistoryTable(data);
    };
};

// Update history table with data
const updateHistoryTable = (data) => {
    const tbody = document.getElementById('historyTable').getElementsByTagName('tbody')[0];
    tbody.innerHTML = '';

    // Sort data by total duration
    const sortedData = data.sort((a, b) => b.totalDuration - a.totalDuration);

    for (const item of sortedData) {
        const row = tbody.insertRow();
        row.insertCell(0).textContent = item.soundType;
        row.insertCell(1).textContent = formatTime(item.totalDuration);
        row.insertCell(2).textContent = item.avgDb.toFixed(1);
        row.insertCell(3).textContent = item.maxDb.toFixed(1);
        row.insertCell(4).textContent = (item.count > 0 ? (item.avgDb / item.count).toFixed(3) : "0.000");
    }
};

// Time formatting helper
function formatTime(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

// Update time displays
function updateTimes() {
    const totalElapsed = appStartTime ? (Date.now() - appStartTime) / 1000 : 0;
    document.getElementById('totalTime').textContent = `⏱️ Total Time: ${formatTime(totalElapsed)}`;
    document.getElementById('quietTime').textContent = `🤫 Quiet Time: ${formatTime(quietTime)}`;
    document.getElementById('noisyTime').textContent = `📢 Noisy Time: ${formatTime(noisyTime)}`;
}

// Update exposure table
function updateExposureTable() {
    const tbody = document.getElementById('exposureTable').getElementsByTagName('tbody')[0];
    tbody.innerHTML = '';
    
    const sortedData = [...exposureData.entries()]
        .sort((a, b) => b[1].totalDuration - a[1].totalDuration);
    
    for (const [soundType, data] of sortedData) {
        const row = tbody.insertRow();
        row.insertCell(0).textContent = soundType;
        row.insertCell(1).textContent = formatTime(data.totalDuration);
        row.insertCell(2).textContent = data.avgDb.toFixed(1);
        row.insertCell(3).textContent = data.maxDb.toFixed(1);
    }
}

// Update detection table
function updateDetectionTable(results, dbLevel) {
    const tbody = document.getElementById('detectionTable').getElementsByTagName('tbody')[0];
    tbody.innerHTML = '';
    
    for (const result of results) {
        const row = tbody.insertRow();
        row.insertCell(0).textContent = `🔹 ${result.className}`;
        row.insertCell(1).textContent = result.score.toFixed(3);
        row.insertCell(2).textContent = dbLevel.toFixed(1);
    }
}

// Update dB level display
function updateDbLevel(dbLevel) {
    const dbElement = document.getElementById('dbLevel');
    let color = 'green';
    if (dbLevel > -20) color = 'red';
    else if (dbLevel > -30) color = 'orange';
    
    dbElement.textContent = `🔊 Sound Level: ${dbLevel.toFixed(1)} dB`;
    dbElement.style.color = color;
}

// Process audio data
async function processAudioData(audioData) {
    const rms = Math.sqrt(audioData.reduce((sum, val) => sum + val * val, 0) / audioData.length);
    const dbLevel = 20 * Math.log10(Math.max(rms, 1e-10));
    
    // Update times
    const now = Date.now();
    const timeDiff = (now - (startTime || now)) / 1000;
    if (dbLevel > DB_THRESHOLD) {
        noisyTime += timeDiff;
    } else {
        quietTime += timeDiff;
        // Clear detection table when silent
        const tbody = document.getElementById('detectionTable').getElementsByTagName('tbody')[0];
        tbody.innerHTML = '';
    }
    startTime = now;
    
    updateDbLevel(dbLevel);
    updateTimes();
    
    if (dbLevel > DB_THRESHOLD) {
        // Prepare audio data for model
        const tensor = tf.tensor1d(audioData);
        
        try {
            // Run model and get predictions
            const [predictions, embeddings, spectrogram] = await model.predict(tensor);
            // Get scores from the predictions tensor (first row since we only have one input)
            const scores = await predictions.slice([0, 0], [1, 521]).dataSync();
            
            // Clean up tensors
            tensor.dispose();
            predictions.dispose();
            embeddings.dispose();
            spectrogram.dispose();
            
            // Get top 5 predictions
            const indices = Array.from(scores)
                .map((score, index) => ({ score, index }))
                .sort((a, b) => b.score - a.score)
                .slice(0, 5);
                
            const results = indices
                .filter(({ score }) => score > 0.3)
                .map(({ score, index }) => ({
                    className: classNames[index],
                    score
                }));
                
            // Update UI with results
            if (results.length > 0) {
                updateDetectionTable(results, dbLevel);
                
                // Update exposure data for the highest scoring sound
                const topResult = results[0];
                const data = exposureData.get(topResult.className) || {
                    totalDuration: 0,
                    avgDb: dbLevel,
                    maxDb: dbLevel,
                    count: 0
                };
                
                data.totalDuration += 1;
                data.avgDb = (data.avgDb * data.count + dbLevel) / (data.count + 1);
                data.maxDb = Math.max(data.maxDb, dbLevel);
                data.count += 1;
                
                exposureData.set(topResult.className, data);
                updateExposureTable();
            }
        } catch (error) {
            console.error('Error processing audio:', error);
        }
    }
}

// Initialize audio capture
async function initAudio() {
    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        audioContext = new AudioContext();
        const source = audioContext.createMediaStreamSource(mediaStream);
        const processor = audioContext.createScriptProcessor(16384, 1, 1);
        
        processor.onaudioprocess = (e) => {
            const audioData = e.inputBuffer.getChannelData(0);
            processAudioData(Array.from(audioData));
        };
        
        source.connect(processor);
        processor.connect(audioContext.destination);
        
        document.getElementById('statusLabel').textContent = 'Microphone active - Waiting for sound...';
        
    } catch (error) {
        console.error('Error initializing audio:', error);
        document.getElementById('statusLabel').textContent = 'Error: Could not access microphone';
    }
}

// Load YAMNet model and class names
async function init() {
    try {
        // Initialize IndexedDB
        initDb();

        // Load model
        model = await tf.loadGraphModel('model/yamnet-tfjs-tfjs-v1/model.json');
        
        // Load class names from local file
        const response = await fetch('model/yamnet-tfjs-tfjs-v1/yamnet_class_map.csv');
        const csvText = await response.text();
        classNames = csvText
            .split('\n')
            .slice(1) // Skip header
            .filter(line => line.trim())
            .map(line => line.split(',')[2].trim());
        
        // Initialize audio capture
        await initAudio();
        
        // Start update timers
        setInterval(updateExposureTable, 1000);
        setInterval(saveExposureData, 10000); // Save to IndexedDB every 10 seconds
        
    } catch (error) {
        console.error('Error initializing:', error);
        document.getElementById('statusLabel').textContent = 'Error: Could not initialize application';
    }
}

// Start initialization when page loads
window.addEventListener('load', init);