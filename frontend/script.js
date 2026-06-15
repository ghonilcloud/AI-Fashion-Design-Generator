// Configuration
const API_BASE_URL = window.API_BASE_URL || "http://localhost:8000";

// State
let uploadedFile = null;
let availableTones = [];
let availableKanseiWords = [];
let designsCreated = 0;

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const sketchInput = document.getElementById('sketchInput');
const previewImage = document.getElementById('previewImage');
const uploadPlaceholder = uploadArea.querySelector('.upload-placeholder');
const generateBtn = document.getElementById('generateBtn');
const tonesContainer = document.getElementById('tonesContainer');
const kanseiWordsContainer = document.getElementById('kanseiWordsContainer');
const promptStrategySelect = document.getElementById('promptStrategy');
const resultsSection = document.getElementById('resultsSection');
const originalSketch = document.getElementById('originalSketch');
const generatedImage = document.getElementById('generatedImage');
const promptDisplay = document.getElementById('promptDisplay');
const notesDisplay = document.getElementById('notesDisplay');
const copyPromptBtn = document.getElementById('copyPromptBtn');
const errorToast = document.getElementById('errorToast');
const errorMessage = document.getElementById('errorMessage');

// Initialize
async function init() {
    await loadTonesAndKansei();
    setupEventListeners();
    initializeSidebar();
    loadUserProfile();
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
        formData.append('prompt_strategy', promptStrategySelect ? promptStrategySelect.value : 'llm');
        
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
        
        // Increment designs count
        incrementDesignsCount();
        
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
    const strategyNote = result.prompt_strategy ? `Prompt strategy: ${result.prompt_strategy}. ` : '';
    notesDisplay.textContent = `${strategyNote}${result.notes || 'No additional notes'}`;
    
    // Show results section
    resultsSection.style.display = 'block';
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
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

// Initialize sidebar
function initializeSidebar() {
    const sidebar = document.getElementById('sidebar');
    const sidebarToggleBtn = document.getElementById('sidebarToggleBtn');
    const sidebarToggle = document.getElementById('sidebarToggle');
    const editProfileBtn = document.getElementById('editProfileBtn');
    const logoutBtn = document.getElementById('logoutBtn');
    
    // Toggle sidebar on mobile
    if (sidebarToggleBtn) {
        sidebarToggleBtn.addEventListener('click', () => {
            sidebar.classList.toggle('active');
        });
    }
    
    // Close sidebar button
    if (sidebarToggle) {
        sidebarToggle.addEventListener('click', () => {
            sidebar.classList.remove('active');
        });
    }
    
    // Edit profile button
    if (editProfileBtn) {
        editProfileBtn.addEventListener('click', handleEditProfile);
    }
    
    // Logout button
    if (logoutBtn) {
        logoutBtn.addEventListener('click', handleLogout);
    }
}

// Load user profile
function loadUserProfile() {
    // Load from localStorage or use default values
    const profile = {
        name: localStorage.getItem('userName') || 'Fashion Designer',
        email: localStorage.getItem('userEmail') || 'user@example.com',
        designsCount: parseInt(localStorage.getItem('designsCount')) || 0,
        memberSince: localStorage.getItem('memberSince') || 'Dec 2025'
    };
    
    // Update UI
    document.getElementById('profileName').textContent = profile.name;
    document.getElementById('profileEmail').textContent = profile.email;
    document.getElementById('designsCount').textContent = profile.designsCount;
    document.getElementById('memberSince').textContent = profile.memberSince;
    
    // Update avatar
    const avatarUrl = `https://ui-avatars.com/api/?name=${encodeURIComponent(profile.name)}&background=6366f1&color=fff&size=120`;
    document.getElementById('profileAvatar').src = avatarUrl;
    
    designsCreated = profile.designsCount;
}

// Handle edit profile
function handleEditProfile() {
    const currentName = document.getElementById('profileName').textContent;
    const currentEmail = document.getElementById('profileEmail').textContent;
    
    const newName = prompt('Enter your name:', currentName);
    if (newName && newName.trim()) {
        const newEmail = prompt('Enter your email:', currentEmail);
        if (newEmail && newEmail.trim()) {
            // Update profile
            document.getElementById('profileName').textContent = newName.trim();
            document.getElementById('profileEmail').textContent = newEmail.trim();
            
            // Update avatar
            const avatarUrl = `https://ui-avatars.com/api/?name=${encodeURIComponent(newName.trim())}&background=6366f1&color=fff&size=120`;
            document.getElementById('profileAvatar').src = avatarUrl;
            
            // Save to localStorage
            localStorage.setItem('userName', newName.trim());
            localStorage.setItem('userEmail', newEmail.trim());
            
            showSuccess('Profile updated successfully!');
        }
    }
}

// Handle logout
function handleLogout() {
    if (confirm('Are you sure you want to logout?')) {
        // Could clear localStorage or redirect to login page
        showSuccess('Logged out successfully!');
        // For demo purposes, just show a message
    }
}

// Update designs count
function incrementDesignsCount() {
    designsCreated++;
    document.getElementById('designsCount').textContent = designsCreated;
    localStorage.setItem('designsCount', designsCreated);
}

// Show success message
function showSuccess(message) {
    // Create a success toast similar to error toast
    const toast = document.createElement('div');
    toast.className = 'toast success-toast';
    toast.style.cssText = 'position: fixed; top: 2rem; right: 2rem; padding: 1rem 1.5rem; border-radius: 8px; background: var(--success-color); color: white; z-index: 1000; animation: slideIn 0.3s ease;';
    toast.textContent = message;
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.remove();
    }, 3000);
}

// Initialize app when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
