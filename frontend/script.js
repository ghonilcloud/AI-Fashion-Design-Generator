// Configuration
const API_BASE_URL = window.API_BASE_URL || "http://localhost:8000";

// State
let uploadedFile = null;
let availableTones = [];
let availableKanseiWords = [];
let lastGenerationSettings = null;

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const sketchInput = document.getElementById('sketchInput');
const previewImage = document.getElementById('previewImage');
const uploadPlaceholder = uploadArea.querySelector('.upload-placeholder');
const generateBtn = document.getElementById('generateBtn');
const tonesContainer = document.getElementById('tonesContainer');
const kanseiWordsContainer = document.getElementById('kanseiWordsContainer');
const promptStrategySelect = document.getElementById('promptStrategy');
const modelKeySelect = document.getElementById('modelKey');
const inferenceStepsSelect = document.getElementById('inferenceSteps');
const guidanceScaleInput = document.getElementById('guidanceScale');
const controlnetScaleSelect = document.getElementById('controlnetScale');
const seedInput = document.getElementById('seedInput');
const resultsSection = document.getElementById('resultsSection');
const originalSketch = document.getElementById('originalSketch');
const generatedImage = document.getElementById('generatedImage');
const promptDisplay = document.getElementById('promptDisplay');
const metadataGrid = document.getElementById('metadataGrid');
const notesDisplay = document.getElementById('notesDisplay');
const copyPromptBtn = document.getElementById('copyPromptBtn');
const errorToast = document.getElementById('errorToast');
const errorMessage = document.getElementById('errorMessage');

// Initialize
async function init() {
    await loadTonesAndKansei();
    setupEventListeners();
}

// Load available tones and Kansei words from API
async function loadTonesAndKansei() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/tones`);
        const data = await response.json();
        
        availableTones = data.tones;
        availableKanseiWords = data.kansei_words;
        
        renderCheckboxes(tonesContainer, availableTones, 'tone');
        renderCheckboxes(kanseiWordsContainer, availableKanseiWords, 'kansei');
    } catch (error) {
        console.error('Error loading tones:', error);
        showError('Failed to load tones and Kansei words. Make sure the backend is running.');
    }
}

// Render checkbox groups
function renderCheckboxes(container, items, name) {
    container.innerHTML = items.map((item, index) => `
        <div class="checkbox-item">
            <input type="checkbox" id="${name}-${index}" name="${name}" value="${item}">
            <label for="${name}-${index}">${item}</label>
        </div>
    `).join('');
    
    // Add tone limit enforcement (max 3)
    if (name === 'tone') {
        const checkboxes = container.querySelectorAll('input[type="checkbox"]');
        checkboxes.forEach(checkbox => {
            checkbox.addEventListener('change', () => {
                const checkedCount = container.querySelectorAll('input[type="checkbox"]:checked').length;
                if (checkedCount >= 3) {
                    // Disable unchecked boxes
                    checkboxes.forEach(cb => {
                        if (!cb.checked) cb.disabled = true;
                    });
                } else {
                    // Enable all boxes
                    checkboxes.forEach(cb => cb.disabled = false);
                }
            });
        });
    }
}

// Setup event listeners
function setupEventListeners() {
    // Upload area click
    uploadArea.addEventListener('click', () => sketchInput.click());
    
    // File input change
    sketchInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });
    
    // Generate button
    generateBtn.addEventListener('click', handleGenerate);
    
    // Copy prompt button
    copyPromptBtn.addEventListener('click', copyPrompt);
}

// Handle file selection
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

// Handle file upload
function handleFile(file) {
    // Validate file type
    if (!file.type.match('image.*')) {
        showError('Please upload an image file (PNG, JPG, or JPEG)');
        return;
    }
    
    // Validate file size (10MB)
    if (file.size > 10 * 1024 * 1024) {
        showError('File size must be less than 10MB');
        return;
    }
    
    uploadedFile = file;
    
    // Preview image
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewImage.style.display = 'block';
        uploadPlaceholder.style.display = 'none';
        generateBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

// Get selected checkboxes
function getSelectedValues(name) {
    const checkboxes = document.querySelectorAll(`input[name="${name}"]:checked`);
    return Array.from(checkboxes).map(cb => cb.value);
}

// Handle generate button click
async function handleGenerate() {
    if (!uploadedFile) {
        showError('Please upload a sketch first');
        return;
    }
    
    // Check if at least one tone or Kansei word is selected
    const selectedTones = getSelectedValues('tone');
    const selectedKanseiWords = getSelectedValues('kansei');
    
    if (selectedTones.length === 0 && selectedKanseiWords.length === 0) {
        showError('Please select at least one tone or Kansei word');
        return;
    }
    
    // Show loading state
    generateBtn.disabled = true;
    generateBtn.querySelector('.btn-text').textContent = 'Generating...';
    generateBtn.querySelector('.loader').style.display = 'inline-block';
    
    try {
        // Prepare form data
        const formData = new FormData();
        formData.append('image', uploadedFile);
        formData.append('kansei_text', 'Fashion design based on selected tones and Kansei words');
        const generationSettings = getAdvancedSettings();
        lastGenerationSettings = generationSettings;

        formData.append('prompt_strategy', generationSettings.prompt_strategy);
        formData.append('model_key', generationSettings.model_key);
        formData.append('num_inference_steps', generationSettings.num_inference_steps);
        formData.append('guidance_scale', generationSettings.guidance_scale);
        formData.append('controlnet_conditioning_scale', generationSettings.controlnet_conditioning_scale);
        formData.append('seed', generationSettings.seed);
        
        // Add selected tones
        const selectedTones = getSelectedValues('tone');
        selectedTones.forEach(tone => formData.append('tones', tone));
        
        // Add selected Kansei words
        const selectedKanseiWords = getSelectedValues('kansei');
        selectedKanseiWords.forEach(word => formData.append('kansei_words', word));
        
        // Send request
        const response = await fetch(`${API_BASE_URL}/api/generate-design`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        
        const result = await response.json();
        
        // Display results
        displayResults(result);
        
    } catch (error) {
        console.error('Generation error:', error);
        showError('Failed to generate design. Please make sure the backend server is running and try again.');
    } finally {
        // Reset button state
        generateBtn.disabled = false;
        generateBtn.querySelector('.btn-text').textContent = 'Generate Design';
        generateBtn.querySelector('.loader').style.display = 'none';
    }
}

// Display results
function displayResults(result) {
    // Show original sketch
    originalSketch.src = previewImage.src;
    
    // Show generated image
    generatedImage.src = `${API_BASE_URL}${result.generated_image_url}`;
    
    // Show prompt
    promptDisplay.textContent = result.llm_prompt;
    
    // Show notes
    const metadata = normalizeGenerationMetadata(result);
    renderGenerationMetadata(metadata);
    notesDisplay.textContent = result.notes || 'No additional notes';
    
    // Show results section
    resultsSection.style.display = 'block';
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function getAdvancedSettings() {
    return {
        prompt_strategy: promptStrategySelect ? promptStrategySelect.value : 'llm',
        model_key: modelKeySelect ? modelKeySelect.value : 'sd_v1_5',
        num_inference_steps: inferenceStepsSelect ? inferenceStepsSelect.value : '20',
        guidance_scale: guidanceScaleInput ? guidanceScaleInput.value || '7.5' : '7.5',
        controlnet_conditioning_scale: controlnetScaleSelect ? controlnetScaleSelect.value : '1.0',
        seed: seedInput ? seedInput.value || '42' : '42'
    };
}

function normalizeGenerationMetadata(result) {
    const metadata = result.generation_metadata || {};
    const fallback = lastGenerationSettings || getAdvancedSettings();

    return {
        model_key: metadata.model_key || fallback.model_key,
        prompt_strategy: result.prompt_strategy || metadata.prompt_strategy || fallback.prompt_strategy,
        num_inference_steps: metadata.num_inference_steps ?? fallback.num_inference_steps,
        guidance_scale: metadata.guidance_scale ?? fallback.guidance_scale,
        controlnet_conditioning_scale: metadata.controlnet_conditioning_scale ?? fallback.controlnet_conditioning_scale,
        seed: metadata.seed ?? fallback.seed
    };
}

function renderGenerationMetadata(metadata) {
    if (!metadataGrid) return;

    const rows = [
        ['Model', formatModelName(metadata.model_key)],
        ['Prompt Strategy', formatPromptStrategy(metadata.prompt_strategy)],
        ['Steps', metadata.num_inference_steps],
        ['CFG Scale', metadata.guidance_scale],
        ['ControlNet Scale', metadata.controlnet_conditioning_scale],
        ['Seed', metadata.seed]
    ];

    metadataGrid.innerHTML = rows.map(([label, value]) => `
        <div class="metadata-item">
            <span class="metadata-label">${label}</span>
            <span class="metadata-value">${value ?? 'N/A'}</span>
        </div>
    `).join('');
}

function formatModelName(modelKey) {
    const names = {
        sd_v1_5: 'SD v1.5',
        sdxl: 'SDXL',
        ssd_1b: 'SSD-1B'
    };
    return names[modelKey] || modelKey || 'N/A';
}

function formatPromptStrategy(strategy) {
    const names = {
        llm: 'LLM Prompt',
        rule_based: 'Rule-Based Prompt'
    };
    return names[strategy] || strategy || 'N/A';
}

// Copy prompt to clipboard
async function copyPrompt() {
    try {
        await navigator.clipboard.writeText(promptDisplay.textContent);
        
        // Visual feedback
        const originalText = copyPromptBtn.textContent;
        copyPromptBtn.textContent = '✓ Copied!';
        copyPromptBtn.style.background = 'var(--success-color)';
        
        setTimeout(() => {
            copyPromptBtn.textContent = originalText;
            copyPromptBtn.style.background = '';
        }, 2000);
    } catch (error) {
        showError('Failed to copy to clipboard');
    }
}

// Show error message
function showError(message) {
    errorMessage.textContent = message;
    errorToast.style.display = 'flex';
    
    setTimeout(hideError, 5000);
}

// Hide error message
function hideError() {
    errorToast.style.display = 'none';
}

// Initialize app when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
