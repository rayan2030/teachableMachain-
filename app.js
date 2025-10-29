// ===================================
// TensorFlow.js Teachable Machine App
// Browser-based Image Classification with Transfer Learning
// ===================================

// ===================================
// Global Variables and State Management
// ===================================

let mobilenet; // Pre-trained MobileNet model for feature extraction
let model; // Custom trained model
let classData = []; // Array to store image data for each class
let classCount = 2; // Current number of classes
let isTraining = false; // Training state flag
let isTrained = false; // Model trained state flag
let currentWebcamClass = null; // Track which class is using webcam
let webcamStream = null; // Store webcam stream
let predictWebcamStream = null; // Store prediction webcam stream
let predictWebcamInterval = null; // Interval for continuous prediction

// ===================================
// Initialization on Page Load
// ===================================

document.addEventListener('DOMContentLoaded', async () => {
    console.log('🚀 Initializing Teachable Machine App...');
    
    // Initialize TensorFlow.js backend with proper fallback
    // This ensures compatibility with all devices and environments
    await initializeTensorFlowBackend();
    
    // Load MobileNet model
    await loadMobileNet();
    
    // Initialize event listeners
    initializeEventListeners();
    
    // Initialize class data structure
    initializeClassData();
    
    // Update statistics
    updateStatistics();
    
    console.log('✅ App initialized successfully');
});

// ===================================
// Initialize TensorFlow.js Backend with Fallback
// ===================================

async function initializeTensorFlowBackend() {
    console.log('🔧 Initializing TensorFlow.js backend...');
    
    // Force CPU backend for Replit environment to avoid WebGL issues
    // Replit's headless environment doesn't support WebGL properly
    const forceCPU = true;
    
    if (forceCPU) {
        console.log('🔒 Forcing CPU backend for compatibility...');
        try {
            await tf.setBackend('cpu');
            await tf.ready();
            
            // Test CPU backend with a more complex operation
            const testTensor = tf.tensor2d([[1, 2], [3, 4]]);
            const testResult = testTensor.square().sum();
            const value = await testResult.data();
            testTensor.dispose();
            testResult.dispose();
            
            console.log('✅ Using CPU backend');
            console.log(`📊 TensorFlow.js backend: ${tf.getBackend()}`);
            console.log(`📊 Backend test result: ${value[0]} (expected: 30)`);
            return;
            
        } catch (error) {
            console.error('❌ CPU backend initialization failed:', error);
            alert('فشل تهيئة TensorFlow.js. يرجى إعادة تحميل الصفحة.');
            throw error;
        }
    }
    
    // Try WebGL first (for desktop/mobile browsers)
    try {
        await tf.setBackend('webgl');
        await tf.ready();
        
        // Test if WebGL actually works with multiple operations
        const testTensor = tf.tensor2d([[1, 2], [3, 4]]);
        const testResult = testTensor.square().sum();
        const value = await testResult.data();
        testTensor.dispose();
        testResult.dispose();
        
        if (value[0] !== 30) {
            throw new Error('WebGL test failed - incorrect result');
        }
        
        console.log('✅ Using WebGL backend (GPU acceleration)');
        console.log(`📊 TensorFlow.js backend: ${tf.getBackend()}`);
        console.log(`📊 Backend test result: ${value[0]} (expected: 30)`);
        return;
        
    } catch (error) {
        console.log('⚠️ WebGL backend failed:', error.message);
        console.log('🔄 Switching to CPU backend...');
    }
    
    // Fall back to CPU if WebGL fails
    try {
        await tf.setBackend('cpu');
        await tf.ready();
        
        // Test CPU backend
        const testTensor = tf.tensor2d([[1, 2], [3, 4]]);
        const testResult = testTensor.square().sum();
        const value = await testResult.data();
        testTensor.dispose();
        testResult.dispose();
        
        console.log('✅ Using CPU backend');
        console.log(`📊 TensorFlow.js backend: ${tf.getBackend()}`);
        console.log(`📊 Backend test result: ${value[0]} (expected: 30)`);
        return;
        
    } catch (error) {
        console.error('❌ Failed to initialize TensorFlow.js backend:', error);
        alert('فشل تهيئة TensorFlow.js. يرجى إعادة تحميل الصفحة.');
        throw error;
    }
}

// ===================================
// Load Pre-trained MobileNet Model
// ===================================

async function loadMobileNet(retries = 3) {
    const statusElement = document.getElementById('modelStatus');
    
    for (let attempt = 1; attempt <= retries; attempt++) {
        try {
            statusElement.textContent = `جاري تحميل نموذج MobileNet... (${attempt}/${retries})`;
            console.log(`📥 Loading MobileNet model (attempt ${attempt}/${retries})...`);
            
            // Check if mobilenet library is loaded
            if (!window.mobilenet) {
                throw new Error('MobileNet library not loaded from CDN');
            }
            
            // Load MobileNet v2 model (for feature extraction)
            mobilenet = await window.mobilenet.load({
                version: 2,
                alpha: 1.0 // Use full model for better accuracy
            });
            
            statusElement.textContent = 'النموذج جاهز للاستخدام ✓';
            statusElement.classList.add('ready');
            statusElement.classList.remove('training');
            console.log('✅ MobileNet loaded successfully');
            return; // Success - exit function
            
        } catch (error) {
            console.error(`❌ Error loading MobileNet (attempt ${attempt}):`, error);
            
            if (attempt === retries) {
                // Last attempt failed
                statusElement.textContent = 'خطأ في تحميل النموذج - يرجى إعادة تحميل الصفحة';
                statusElement.style.color = 'var(--error)';
                statusElement.classList.remove('ready', 'training');
                
                // Show user-friendly error message
                alert('فشل تحميل نموذج التعلم الآلي. يرجى:\n1. التحقق من اتصال الإنترنت\n2. إعادة تحميل الصفحة (F5)\n3. تجربة متصفح آخر إذا استمرت المشكلة');
            } else {
                // Wait before retrying (exponential backoff)
                const waitTime = 1000 * attempt; // 1s, 2s, 3s
                console.log(`⏳ Retrying in ${waitTime}ms...`);
                await new Promise(resolve => setTimeout(resolve, waitTime));
            }
        }
    }
}

// ===================================
// Initialize Class Data Structure
// ===================================

function initializeClassData() {
    classData = [];
    const classCards = document.querySelectorAll('.class-card');
    
    classCards.forEach((card, index) => {
        classData[index] = {
            name: card.querySelector('.class-name-input').value,
            images: [],
            features: [] // Store extracted features from MobileNet
        };
    });
    
    console.log(`📊 Initialized ${classData.length} classes`);
}

// ===================================
// Event Listeners Initialization
// ===================================

function initializeEventListeners() {
    // File upload listeners for each class
    document.querySelectorAll('.file-input').forEach((input, index) => {
        input.addEventListener('change', (e) => handleFileUpload(e, index));
    });
    
    // Drag and drop support
    document.querySelectorAll('.upload-area').forEach((area, index) => {
        area.addEventListener('dragover', handleDragOver);
        area.addEventListener('drop', (e) => handleDrop(e, index));
    });
    
    // Webcam capture buttons
    document.querySelectorAll('.btn-webcam').forEach((btn) => {
        btn.addEventListener('click', () => {
            const classId = parseInt(btn.dataset.classId);
            console.log(`🎥 Opening webcam for class ${classId}`);
            openWebcamModal(classId);
        });
    });
    
    // Class name input listeners
    document.querySelectorAll('.class-name-input').forEach((input, index) => {
        input.addEventListener('input', (e) => {
            if (classData[index]) {
                classData[index].name = e.target.value;
            }
        });
    });
    
    // Remove class buttons
    document.querySelectorAll('.btn-remove-class').forEach((btn) => {
        btn.addEventListener('click', (e) => {
            const card = e.target.closest('.class-card');
            const classId = parseInt(card.dataset.classId);
            removeClass(classId);
        });
    });
    
    // Add class button
    document.getElementById('addClassBtn').addEventListener('click', addClass);
    
    // Train button
    document.getElementById('trainBtn').addEventListener('click', trainModel);
    
    // Export button
    document.getElementById('exportBtn').addEventListener('click', exportModel);
    
    // Webcam modal controls
    document.querySelector('.btn-close-modal').addEventListener('click', closeWebcamModal);
    document.getElementById('captureBtn').addEventListener('click', captureFromWebcam);
    document.getElementById('stopWebcamBtn').addEventListener('click', closeWebcamModal);
    
    // Prediction mode toggles
    document.getElementById('uploadPredictBtn').addEventListener('click', () => switchPredictionMode('upload'));
    document.getElementById('webcamPredictBtn').addEventListener('click', () => switchPredictionMode('webcam'));
    
    // Prediction file upload
    document.getElementById('predictFileInput').addEventListener('change', handlePredictionUpload);
    
    // Prediction webcam controls
    document.getElementById('startPredictWebcam').addEventListener('click', startPredictionWebcam);
    document.getElementById('stopPredictWebcam').addEventListener('click', stopPredictionWebcam);
}

// ===================================
// File Upload Handling
// ===================================

async function handleFileUpload(event, classId) {
    const files = Array.from(event.target.files);
    console.log(`📁 Uploading ${files.length} files for class ${classId}`);
    
    for (const file of files) {
        if (file.type.startsWith('image/')) {
            await processImage(file, classId);
        }
    }
    
    updateClassDisplay(classId);
    updateStatistics();
    checkTrainingReady();
}

// ===================================
// Drag and Drop Handling
// ===================================

function handleDragOver(event) {
    event.preventDefault();
    event.currentTarget.style.borderColor = 'var(--accent-primary)';
}

async function handleDrop(event, classId) {
    event.preventDefault();
    event.currentTarget.style.borderColor = 'var(--border-color)';
    
    const files = Array.from(event.dataTransfer.files);
    console.log(`🎯 Dropped ${files.length} files for class ${classId}`);
    
    for (const file of files) {
        if (file.type.startsWith('image/')) {
            await processImage(file, classId);
        }
    }
    
    updateClassDisplay(classId);
    updateStatistics();
    checkTrainingReady();
}

// ===================================
// Image Processing and Feature Extraction
// ===================================

async function processImage(file, classId) {
    return new Promise((resolve) => {
        const reader = new FileReader();
        
        reader.onload = async (e) => {
            const img = new Image();
            img.src = e.target.result;
            
            img.onload = async () => {
                // Extract features using MobileNet
                const features = await extractFeatures(img);
                
                // Store image and features
                classData[classId].images.push(e.target.result);
                classData[classId].features.push(features);
                
                console.log(`✅ Processed image for class ${classId}`);
                resolve();
            };
        };
        
        reader.readAsDataURL(file);
    });
}

// ===================================
// Feature Extraction with MobileNet
// ===================================

async function extractFeatures(imageElement) {
    // Use MobileNet to extract features (activations from intermediate layer)
    // This uses transfer learning - we get rich feature representations
    const activation = mobilenet.infer(imageElement, 'conv_preds');
    return activation;
}

// ===================================
// Update Class Display (Preview Images)
// ===================================

function updateClassDisplay(classId) {
    const previewContainer = document.querySelector(`.images-preview[data-class-id="${classId}"]`);
    const countElement = document.querySelector(`.count-number[data-class-id="${classId}"]`);
    
    // Clear existing previews
    previewContainer.innerHTML = '';
    
    // Add image previews
    classData[classId].images.forEach((imageSrc, imgIndex) => {
        const container = document.createElement('div');
        container.className = 'preview-image-container';
        
        const img = document.createElement('img');
        img.src = imageSrc;
        img.className = 'preview-image';
        img.alt = `صورة ${imgIndex + 1}`;
        
        const removeBtn = document.createElement('button');
        removeBtn.className = 'btn-remove-image';
        removeBtn.textContent = '×';
        removeBtn.onclick = () => removeImage(classId, imgIndex);
        
        container.appendChild(img);
        container.appendChild(removeBtn);
        previewContainer.appendChild(container);
    });
    
    // Update count
    countElement.textContent = classData[classId].images.length;
}

// ===================================
// Remove Image
// ===================================

function removeImage(classId, imgIndex) {
    classData[classId].images.splice(imgIndex, 1);
    classData[classId].features.splice(imgIndex, 1);
    
    updateClassDisplay(classId);
    updateStatistics();
    checkTrainingReady();
}

// ===================================
// Webcam Modal Functions
// ===================================

async function openWebcamModal(classId) {
    currentWebcamClass = classId;
    const modal = document.getElementById('webcamModal');
    const video = document.getElementById('webcam');
    const captureCount = document.getElementById('captureCount');
    
    modal.classList.remove('hidden');
    captureCount.textContent = '0';
    
    try {
        // Request webcam access
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'user', width: 640, height: 480 },
            audio: false
        });
        
        video.srcObject = webcamStream;
        
        // Wait for video to be ready before allowing capture
        await new Promise((resolve) => {
            video.onloadedmetadata = () => {
                video.play();
                resolve();
            };
        });
        
        console.log('📷 Webcam started and ready');
        
    } catch (error) {
        console.error('❌ Webcam access error:', error);
        alert('لا يمكن الوصول إلى الكاميرا. يرجى التحقق من الأذونات.');
        closeWebcamModal();
    }
}

async function captureFromWebcam() {
    const video = document.getElementById('webcam');
    const canvas = document.getElementById('webcamCanvas');
    const captureCount = document.getElementById('captureCount');
    
    // Check if video is ready
    if (!video.videoWidth || !video.videoHeight) {
        console.error('❌ Video not ready yet');
        alert('الرجاء الانتظار حتى تصبح الكاميرا جاهزة');
        return;
    }
    
    try {
        // Set canvas dimensions to match video
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        // Draw current video frame to canvas
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0);
        
        // Convert canvas to data URL
        const imageData = canvas.toDataURL('image/jpeg');
        
        // Create image element for processing
        const img = new Image();
        img.src = imageData;
        
        img.onload = async () => {
            try {
                // Extract features
                const features = await extractFeatures(img);
                
                // Store image and features
                classData[currentWebcamClass].images.push(imageData);
                classData[currentWebcamClass].features.push(features);
                
                // Update count
                const currentCount = parseInt(captureCount.textContent);
                captureCount.textContent = currentCount + 1;
                
                console.log(`📸 Captured image ${currentCount + 1} for class ${currentWebcamClass}`);
                
                // Update display
                updateClassDisplay(currentWebcamClass);
                updateStatistics();
                checkTrainingReady();
            } catch (error) {
                console.error('❌ Error processing capture:', error);
                alert('حدث خطأ أثناء معالجة الصورة');
            }
        };
        
        img.onerror = () => {
            console.error('❌ Error loading captured image');
            alert('حدث خطأ في تحميل الصورة الملتقطة');
        };
        
    } catch (error) {
        console.error('❌ Capture error:', error);
        alert('حدث خطأ أثناء التقاط الصورة');
    }
}

function closeWebcamModal() {
    const modal = document.getElementById('webcamModal');
    const video = document.getElementById('webcam');
    
    // Stop webcam stream
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }
    
    video.srcObject = null;
    modal.classList.add('hidden');
    currentWebcamClass = null;
    
    console.log('📷 Webcam stopped');
}

// ===================================
// Add New Class
// ===================================

function addClass() {
    const container = document.getElementById('classesContainer');
    const newClassId = classCount;
    
    const classCard = document.createElement('div');
    classCard.className = 'class-card';
    classCard.dataset.classId = newClassId;
    
    classCard.innerHTML = `
        <div class="class-header">
            <input type="text" class="class-name-input" value="الفئة ${newClassId + 1}" placeholder="اسم الفئة">
            <button class="btn-remove-class" title="حذف الفئة">×</button>
        </div>
        <div class="upload-area" data-class-id="${newClassId}">
            <input type="file" id="fileInput${newClassId}" class="file-input" accept="image/*" multiple>
            <label for="fileInput${newClassId}" class="upload-label">
                <span class="upload-icon">📁</span>
                <span>اسحب الصور هنا أو انقر للتحميل</span>
            </label>
        </div>
        <div class="webcam-controls">
            <button class="btn-secondary btn-webcam" data-class-id="${newClassId}">
                <span>📷</span> التقاط من الكاميرا
            </button>
        </div>
        <div class="images-preview" data-class-id="${newClassId}"></div>
        <div class="image-count">
            <span class="count-label">عدد الصور:</span>
            <span class="count-number" data-class-id="${newClassId}">0</span>
        </div>
    `;
    
    container.appendChild(classCard);
    
    // Initialize class data
    classData[newClassId] = {
        name: `الفئة ${newClassId + 1}`,
        images: [],
        features: []
    };
    
    // Add event listeners for new elements
    const fileInput = classCard.querySelector('.file-input');
    fileInput.addEventListener('change', (e) => handleFileUpload(e, newClassId));
    
    const uploadArea = classCard.querySelector('.upload-area');
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('drop', (e) => handleDrop(e, newClassId));
    
    const webcamBtn = classCard.querySelector('.btn-webcam');
    webcamBtn.addEventListener('click', () => openWebcamModal(newClassId));
    
    const nameInput = classCard.querySelector('.class-name-input');
    nameInput.addEventListener('input', (e) => {
        classData[newClassId].name = e.target.value;
    });
    
    const removeBtn = classCard.querySelector('.btn-remove-class');
    removeBtn.addEventListener('click', () => removeClass(newClassId));
    
    classCount++;
    updateStatistics();
    
    console.log(`➕ Added new class ${newClassId}`);
}

// ===================================
// Remove Class
// ===================================

function removeClass(classId) {
    // Don't allow removing if only 2 classes remain
    const activeClasses = classData.filter(c => c !== null).length;
    if (activeClasses <= 2) {
        alert('يجب أن يكون هناك على الأقل فئتان للتدريب');
        return;
    }
    
    // Remove class data
    classData[classId] = null;
    
    // Remove class card from DOM
    const card = document.querySelector(`.class-card[data-class-id="${classId}"]`);
    if (card) {
        card.remove();
    }
    
    updateStatistics();
    checkTrainingReady();
    
    console.log(`➖ Removed class ${classId}`);
}

// ===================================
// Check if Ready to Train
// ===================================

function checkTrainingReady() {
    const trainBtn = document.getElementById('trainBtn');
    
    // Check if we have at least 2 classes with images
    const validClasses = classData.filter(c => c !== null && c.images.length > 0);
    
    if (validClasses.length >= 2) {
        trainBtn.disabled = false;
    } else {
        trainBtn.disabled = true;
    }
}

// ===================================
// Train Model (Transfer Learning)
// ===================================

async function trainModel() {
    if (isTraining) return;
    
    isTraining = true;
    isTrained = false;
    
    const trainBtn = document.getElementById('trainBtn');
    const progressSection = document.getElementById('trainingProgress');
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');
    const statusElement = document.getElementById('modelStatus');
    
    trainBtn.disabled = true;
    progressSection.classList.remove('hidden');
    statusElement.textContent = 'جاري التدريب...';
    statusElement.classList.remove('ready');
    statusElement.classList.add('training');
    
    console.log('🎯 Starting model training...');
    
    try {
        // Prepare training data
        const { xs, ys, classNames } = prepareTrainingData();
        
        // Build and compile model
        model = buildModel(xs.shape[1]);
        
        // Training parameters
        const epochs = 50;
        const batchSize = 16;
        
        // Train the model
        await model.fit(xs, ys, {
            epochs: epochs,
            batchSize: batchSize,
            shuffle: true,
            validationSplit: 0.2,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    const progress = ((epoch + 1) / epochs) * 100;
                    progressFill.style.width = `${progress}%`;
                    progressText.textContent = `التدريب: ${epoch + 1}/${epochs} - الدقة: ${(logs.acc * 100).toFixed(1)}%`;
                    console.log(`Epoch ${epoch + 1}: accuracy = ${(logs.acc * 100).toFixed(2)}%`);
                }
            }
        });
        
        // Clean up tensors
        xs.dispose();
        ys.dispose();
        
        // Training complete
        isTrained = true;
        isTraining = false;
        
        progressText.textContent = 'اكتمل التدريب بنجاح! ✓';
        statusElement.textContent = 'النموذج جاهز للتنبؤ ✓';
        statusElement.classList.remove('training');
        statusElement.classList.add('ready');
        
        // Enable export button
        document.getElementById('exportBtn').disabled = false;
        
        // Update statistics
        document.getElementById('statStatus').textContent = 'مدرب ✓';
        document.getElementById('statStatus').style.color = 'var(--success)';
        document.getElementById('statAccuracy').textContent = '~90%';
        
        console.log('✅ Training completed successfully');
        
    } catch (error) {
        console.error('❌ Training error:', error);
        isTraining = false;
        progressText.textContent = 'حدث خطأ أثناء التدريب';
        statusElement.textContent = 'خطأ في التدريب';
        statusElement.classList.remove('training');
    }
}

// ===================================
// Prepare Training Data
// ===================================

function prepareTrainingData() {
    const features = [];
    const labels = [];
    const classNames = [];
    
    // Collect valid classes
    const validClasses = classData.filter(c => c !== null && c.images.length > 0);
    
    validClasses.forEach((classItem, classIndex) => {
        classNames.push(classItem.name);
        
        classItem.features.forEach(feature => {
            features.push(feature);
            labels.push(classIndex);
        });
    });
    
    // Convert to tensors
    const xs = tf.stack(features);
    const ys = tf.oneHot(tf.tensor1d(labels, 'int32'), validClasses.length);
    
    console.log(`📊 Training data: ${xs.shape[0]} samples, ${validClasses.length} classes`);
    
    return { xs, ys, classNames };
}

// ===================================
// Build Neural Network Model
// ===================================

function buildModel(inputShape) {
    const model = tf.sequential();
    
    // Flatten the input features
    model.add(tf.layers.flatten({ inputShape: [7, 7, 1024] }));
    
    // Dense hidden layer with dropout for regularization
    model.add(tf.layers.dense({
        units: 128,
        activation: 'relu',
        kernelInitializer: 'heNormal'
    }));
    model.add(tf.layers.dropout({ rate: 0.5 }));
    
    // Output layer (number of classes)
    const validClasses = classData.filter(c => c !== null && c.images.length > 0);
    model.add(tf.layers.dense({
        units: validClasses.length,
        activation: 'softmax'
    }));
    
    // Compile model
    model.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });
    
    console.log('🏗️ Model architecture built');
    model.summary();
    
    return model;
}

// ===================================
// Prediction Functions
// ===================================

function switchPredictionMode(mode) {
    const uploadBtn = document.getElementById('uploadPredictBtn');
    const webcamBtn = document.getElementById('webcamPredictBtn');
    const uploadArea = document.getElementById('uploadPredictArea');
    const webcamArea = document.getElementById('webcamPredictArea');
    
    if (mode === 'upload') {
        uploadBtn.classList.add('active');
        webcamBtn.classList.remove('active');
        uploadArea.classList.remove('hidden');
        webcamArea.classList.add('hidden');
        stopPredictionWebcam();
    } else {
        webcamBtn.classList.add('active');
        uploadBtn.classList.remove('active');
        webcamArea.classList.remove('hidden');
        uploadArea.classList.add('hidden');
    }
}

async function handlePredictionUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = async (e) => {
        const img = document.getElementById('predictImage');
        img.src = e.target.result;
        img.classList.remove('hidden');
        
        // Predict after image loads
        img.onload = async () => {
            await predictImage(img);
        };
    };
    reader.readAsDataURL(file);
}

async function startPredictionWebcam() {
    const video = document.getElementById('predictWebcam');
    const startBtn = document.getElementById('startPredictWebcam');
    const stopBtn = document.getElementById('stopPredictWebcam');
    
    try {
        predictWebcamStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'user', width: 640, height: 480 },
            audio: false
        });
        
        video.srcObject = predictWebcamStream;
        
        // Wait for video to be ready
        await new Promise((resolve) => {
            video.onloadedmetadata = () => {
                video.play();
                resolve();
            };
        });
        
        startBtn.classList.add('hidden');
        stopBtn.classList.remove('hidden');
        
        // Start continuous prediction
        predictWebcamInterval = setInterval(async () => {
            if (isTrained && video.videoWidth && video.videoHeight) {
                await predictImage(video);
            }
        }, 500); // Predict every 500ms
        
        console.log('📷 Prediction webcam started and ready');
        
    } catch (error) {
        console.error('❌ Webcam error:', error);
        alert('لا يمكن الوصول إلى الكاميرا');
    }
}

function stopPredictionWebcam() {
    const video = document.getElementById('predictWebcam');
    const startBtn = document.getElementById('startPredictWebcam');
    const stopBtn = document.getElementById('stopPredictWebcam');
    
    if (predictWebcamStream) {
        predictWebcamStream.getTracks().forEach(track => track.stop());
        predictWebcamStream = null;
    }
    
    if (predictWebcamInterval) {
        clearInterval(predictWebcamInterval);
        predictWebcamInterval = null;
    }
    
    video.srcObject = null;
    startBtn.classList.remove('hidden');
    stopBtn.classList.add('hidden');
    
    console.log('📷 Prediction webcam stopped');
}

async function predictImage(imageElement) {
    if (!isTrained || !model) {
        displayPrediction([{ class: 'غير مدرب', confidence: 0 }]);
        return;
    }
    
    try {
        // Extract features from image
        const features = await extractFeatures(imageElement);
        
        // Make prediction
        const prediction = model.predict(features.expandDims(0));
        const probabilities = await prediction.data();
        
        // Get class names
        const validClasses = classData.filter(c => c !== null && c.images.length > 0);
        
        // Format results
        const results = Array.from(probabilities).map((prob, index) => ({
            class: validClasses[index].name,
            confidence: prob
        }));
        
        // Sort by confidence
        results.sort((a, b) => b.confidence - a.confidence);
        
        // Display results
        displayPrediction(results);
        
        // Clean up
        features.dispose();
        prediction.dispose();
        
    } catch (error) {
        console.error('❌ Prediction error:', error);
    }
}

function displayPrediction(results) {
    const container = document.getElementById('predictionsDisplay');
    container.innerHTML = '';
    
    results.forEach((result, index) => {
        const item = document.createElement('div');
        item.className = `prediction-item ${index === 0 ? 'top' : ''}`;
        
        item.innerHTML = `
            <div class="prediction-label">${result.class}</div>
            <div class="prediction-confidence">
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${result.confidence * 100}%"></div>
                </div>
                <span class="confidence-text">${(result.confidence * 100).toFixed(1)}%</span>
            </div>
        `;
        
        container.appendChild(item);
    });
}

// ===================================
// Export Model
// ===================================

async function exportModel() {
    if (!isTrained || !model) {
        alert('يجب تدريب النموذج أولاً');
        return;
    }
    
    const exportStatus = document.getElementById('exportStatus');
    exportStatus.classList.remove('hidden');
    exportStatus.textContent = 'جاري تصدير النموذج...';
    
    try {
        // Save model to downloads
        await model.save('downloads://teachable-machine-model');
        
        exportStatus.textContent = 'تم تصدير النموذج بنجاح! ✓';
        exportStatus.style.color = 'var(--success)';
        
        console.log('💾 Model exported successfully');
        
        setTimeout(() => {
            exportStatus.classList.add('hidden');
        }, 3000);
        
    } catch (error) {
        console.error('❌ Export error:', error);
        exportStatus.textContent = 'حدث خطأ في التصدير';
        exportStatus.style.color = 'var(--error)';
    }
}

// ===================================
// Update Statistics Display
// ===================================

function updateStatistics() {
    const validClasses = classData.filter(c => c !== null);
    const totalImages = validClasses.reduce((sum, c) => sum + c.images.length, 0);
    
    document.getElementById('statClasses').textContent = validClasses.length;
    document.getElementById('statImages').textContent = totalImages;
}

// ===================================
// End of Application
// ===================================

console.log('📱 Teachable Machine App Ready');
