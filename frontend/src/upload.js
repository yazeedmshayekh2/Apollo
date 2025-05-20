/**
 * Upload page JavaScript for DocumentScan AI - OCR Application
 * Handles form submission, file previews, and results display
 */

// API configuration
const API_CONFIG = {
  baseUrl: window.location.hostname.includes('ngrok-free.app') 
    ? window.location.origin 
    : (window.location.hostname === 'localhost' ? 'http://localhost:8000' : 'http://'+window.location.hostname+':8000'),
  endpoints: {
    process: '/api/process',
    jobStatus: '/api/jobs/',
    extractedJson: '/api/jobs/' // Will be completed with job_id/json
  },
  pollInterval: 2000,  // How often to check job status (ms)
};

// Debug mode - set to false to always use real API calls
const USE_SIMULATION = false;

document.addEventListener('DOMContentLoaded', () => {
  // Initialize the upload page
  initUploadPage();
  console.log('API base URL:', API_CONFIG.baseUrl);
});

/**
 * Initialize the upload page
 */
function initUploadPage() {
  // Set up file preview handlers
  setupFilePreview('front-input', 'front-preview');
  setupFilePreview('back-input', 'back-preview');
  
  // Set up form submission
  setupFormSubmission();
  
  // Set up result tabs
  setupResultTabs();
  
  // Set up action buttons
  setupActionButtons();
  
  console.log('Upload page initialized');
}

/**
 * Set up file preview functionality
 * @param {string} inputId - ID of the file input element
 * @param {string} previewId - ID of the preview image element
 */
function setupFilePreview(inputId, previewId) {
  const fileInput = document.getElementById(inputId);
  const previewImg = document.getElementById(previewId);
  const placeholder = previewImg.parentElement.querySelector('.upload-placeholder');
  
  if (!fileInput || !previewImg) return;
  
  fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    
    if (file) {
      const reader = new FileReader();
      
      reader.onload = (event) => {
        previewImg.src = event.target.result;
        previewImg.style.display = 'block';
        placeholder.style.display = 'none';
      };
      
      reader.readAsDataURL(file);
    } else {
      previewImg.src = '';
      previewImg.style.display = 'none';
      placeholder.style.display = 'flex';
    }
  });
}

/**
 * Set up form submission
 */
function setupFormSubmission() {
  const form = document.getElementById('upload-form');
  const submitBtn = document.getElementById('submit-btn');
  
  if (!form || !submitBtn) return;
  
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Disable submit button and show loading state
    submitBtn.disabled = true;
    submitBtn.textContent = 'Processing...';
    
    try {
      // Get form data
      const formData = new FormData(form);
      
      // Remove back image if not provided
      if (!formData.get('back') || formData.get('back').size === 0) {
        formData.delete('back');
      }
      
      // Submit the form
      const jobData = await submitForm(formData);
      
      // Show results section
      showResultsSection();
      
      // Start polling for job status
      startJobStatusPolling(jobData.job_id);
    } catch (error) {
      console.error('Error submitting form:', error);
      
      // Show error in results section
      showResultsSection();
      updateStatusIndicator('error', `Error: ${error.message || 'Failed to process form'}`);
    } finally {
      // Re-enable submit button
      submitBtn.disabled = false;
      submitBtn.textContent = 'Process Document';
    }
  });
}

/**
 * Submit the form to the API
 * @param {FormData} formData - Form data to submit
 * @returns {Promise<Object>} - Job data
 */
async function submitForm(formData) {
  try {
    // Debug log all form data being sent to the API
    console.log('Form data being sent to API:');
    for (const [key, value] of formData.entries()) {
      console.log(`${key}: ${value instanceof File ? value.name : value}`);
    }
    
    // Always generate a random user ID since the field is no longer in the UI
    const userId = 'user_' + Math.random().toString(36).substr(2, 9);
    console.log(`Automatically generated user_id: ${userId}`);
    formData.set('user_id', userId);
    
    // In development, we'll simulate the API response if USE_SIMULATION is true
    if (USE_SIMULATION) {
      console.log('Simulation mode: simulating API response');
      
      // Log the form data
      for (const [key, value] of formData.entries()) {
        console.log(`${key}: ${value instanceof File ? value.name : value}`);
      }
      
      // Return simulated job data
      return {
        job_id: 'sim_' + Math.random().toString(36).substr(2, 9),
        status: 'queued',
        created_at: new Date().toISOString()
      };
    }
    
    // Make the actual API call
    console.log('Submitting form to:', `${API_CONFIG.baseUrl}${API_CONFIG.endpoints.process}`);
    
    const response = await fetch(`${API_CONFIG.baseUrl}${API_CONFIG.endpoints.process}`, {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || `HTTP error ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error submitting form:', error);
    throw error;
  }
}

/**
 * Start polling for job status
 * @param {string} jobId - Job ID to poll for
 */
function startJobStatusPolling(jobId) {
  // Update status indicator
  updateStatusIndicator('loading', 'Processing your document...');
  updateProgressBar(10);
  
  // Simulated progress for development
  if (USE_SIMULATION) {
    simulateJobProgress(jobId);
    return;
  }
  
  // Start polling
  const pollInterval = setInterval(async () => {
    try {
      const jobStatus = await checkJobStatus(jobId);
      
      updateStatusDisplay(jobStatus);
      
      if (jobStatus.status === 'completed' || jobStatus.status === 'failed') {
        clearInterval(pollInterval);
      }
    } catch (error) {
      console.error('Error polling job status:', error);
      updateStatusIndicator('error', 'Error checking job status');
      clearInterval(pollInterval);
    }
  }, API_CONFIG.pollInterval);
}

/**
 * Check the status of a job
 * @param {string} jobId - Job ID to check
 * @returns {Promise<Object>} - Job status data
 */
async function checkJobStatus(jobId) {
  const response = await fetch(`${API_CONFIG.baseUrl}${API_CONFIG.endpoints.jobStatus}${jobId}`);
  
  if (!response.ok) {
    throw new Error(`HTTP error ${response.status}`);
  }
  
  return await response.json();
}

/**
 * Update the status display based on job status
 * @param {Object} jobStatus - Job status data
 */
function updateStatusDisplay(jobStatus) {
  console.log('Job status update:', jobStatus);
  
  // Update document type if available
  if (jobStatus.document_type) {
    updateDocumentType(jobStatus.document_type);
  }
  
  switch (jobStatus.status) {
    case 'pending':
      updateStatusIndicator('loading', 'Job queued, waiting to start...');
      updateProgressBar(20);
      break;
    case 'processing':
      // Check if there's more detailed progress info
      if (jobStatus.processing_stage) {
        let message = 'Processing document...';
        let progress = 50;
        
        // Customize message and progress based on stage
        switch(jobStatus.processing_stage) {
          case 'starting':
            message = 'Initializing processing...';
            progress = 30;
            break;
          case 'detecting_type':
            message = 'Detecting document type...';
            progress = 40;
            break;
          case 'processing':
            message = jobStatus.processing_message || 'Processing document...';
            progress = 60;
            break;
          case 'finalizing':
            message = 'Finalizing results...';
            progress = 80;
            break;
        }
        
        updateStatusIndicator('loading', message);
        updateProgressBar(progress);
      } else {
        updateStatusIndicator('loading', 'Processing document...');
        updateProgressBar(50);
      }
      break;
    case 'completed':
      // Check if results were saved to MongoDB
      const wasSavedToMongoDB = jobStatus.user_id && jobStatus.user_id.trim() !== '';
      
      if (wasSavedToMongoDB) {
        updateStatusIndicator('success', 'Processing completed successfully and saved to MongoDB');
      } else {
      updateStatusIndicator('success', 'Processing completed successfully');
      }
      updateProgressBar(100);
      
      // Display the results
      displayResults(jobStatus);
      break;
    case 'completed_with_warnings':
      // Check if results were saved to MongoDB
      const savedWithWarnings = jobStatus.user_id && jobStatus.user_id.trim() !== '';
      
      if (savedWithWarnings) {
        updateStatusIndicator('warning', 'Completed with some missing information and saved to MongoDB');
      } else {
      updateStatusIndicator('warning', 'Completed with some missing information');
      }
      updateProgressBar(100);
      
      // Display the results
      displayResults(jobStatus);
      break;
    case 'failed':
      updateStatusIndicator('error', `Processing failed: ${jobStatus.error || 'Unknown error'}`);
      updateProgressBar(100);
      break;
    default:
      updateStatusIndicator('loading', `Status: ${jobStatus.status}`);
      updateProgressBar(40);
  }
}

/**
 * Update the document type display
 * @param {string} documentType - Document type
 */
function updateDocumentType(documentType) {
  const container = document.getElementById('document-type-container');
  const value = document.getElementById('document-type-value');
  const badge = document.getElementById('document-type-badge');
  
  if (!container || !value || !badge) return;
  
  // Show the container
  container.style.display = 'flex';
  
  // Format the document type for display
  let displayType = 'Unknown';
  let badgeClass = 'unknown';
  
  switch (documentType) {
    case 'vehicle_registration':
      displayType = 'Vehicle Registration Card';
      badgeClass = 'vehicle';
      break;
    case 'id_card_front':
      displayType = 'ID Card (Front)';
      badgeClass = 'id-card';
      break;
    case 'id_card_back':
      displayType = 'ID Card (Back)';
      badgeClass = 'id-card';
      break;
    case 'id_card':
      displayType = 'ID Card (Complete)';
      badgeClass = 'id-card';
      break;
    default:
      if (documentType) {
        displayType = documentType.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
      }
  }
  
  // Update the display
  value.textContent = displayType;
  badge.textContent = displayType.split(' ')[0].toUpperCase();
  badge.className = 'doc-type-badge ' + badgeClass;
}

/**
 * Update the status indicator
 * @param {string} status - Status type ('loading', 'success', 'error', 'warning')
 * @param {string} message - Status message
 */
function updateStatusIndicator(status, message) {
  const statusIcon = document.getElementById('status-icon');
  const statusText = document.getElementById('status-text');
  
  if (!statusIcon || !statusText) return;
  
  // Update icon class based on status
  statusIcon.className = 'status-icon ' + status;
  
  // Update status text
  statusText.textContent = message;
}

/**
 * Update the progress bar
 * @param {number} percent - Progress percentage (0-100)
 */
function updateProgressBar(percent) {
  const progressFill = document.getElementById('progress-fill');
  
  if (!progressFill) return;
  
  progressFill.style.width = `${percent}%`;
}

/**
 * Display the results
 * @param {Object} jobData - Job data containing results
 */
function displayResults(jobData) {
  // Get result containers
  const vehicleData = document.getElementById('vehicle-data');
  const ownerData = document.getElementById('owner-data');
  const registrationData = document.getElementById('registration-data');
  const personalData = document.getElementById('personal-data');
  const extractedJson = document.getElementById('extracted-json');
  const rawData = document.getElementById('raw-data');
  
  if (!vehicleData || !ownerData || !registrationData || !personalData || !extractedJson || !rawData) {
    return;
  }
  
  // Update document type display
  if (jobData.document_type) {
    updateDocumentType(jobData.document_type);
    
    // Update tab visibility based on document type
    const personalTab = document.querySelector('.tab-btn[data-tab="personal"]');
    const faceTab = document.querySelector('.tab-btn[data-tab="face"]');
    const vehicleTab = document.querySelector('.tab-btn[data-tab="vehicle"]');
    const ownerTab = document.querySelector('.tab-btn[data-tab="owner"]');
    const registrationTab = document.querySelector('.tab-btn[data-tab="registration"]');
    
    if (jobData.document_type.startsWith('id_card')) {
      // For ID cards, show personal and face tabs, hide vehicle tabs
      if (personalTab) personalTab.style.display = 'inline-block';
      if (faceTab) faceTab.style.display = 'inline-block';
      
      // Hide vehicle-specific tabs
      if (vehicleTab) vehicleTab.style.display = 'none';
      if (ownerTab) ownerTab.style.display = 'none';
      if (registrationTab) registrationTab.style.display = 'none';
      
      // Switch to personal tab if vehicle tab was active
      if (vehicleTab && vehicleTab.classList.contains('active')) {
        vehicleTab.classList.remove('active');
        if (personalTab) {
          personalTab.classList.add('active');
          
          // Also activate corresponding content
          document.getElementById('vehicle-tab').classList.remove('active');
          document.getElementById('personal-tab').classList.add('active');
        }
      }
    } else {
      // For vehicle registrations, show vehicle tabs, hide ID card tabs
      if (personalTab) personalTab.style.display = 'none';
      if (faceTab) faceTab.style.display = 'none';
      
      // Show vehicle-specific tabs
      if (vehicleTab) vehicleTab.style.display = 'inline-block';
      if (ownerTab) ownerTab.style.display = 'inline-block';
      if (registrationTab) registrationTab.style.display = 'inline-block';
    }
  }
  
  // Process face detection results if available
  if (jobData.face_detection) {
    displayFaceDetectionResults(jobData.face_detection);
  }
  
  // Clear existing content
  vehicleData.innerHTML = '';
  ownerData.innerHTML = '';
  registrationData.innerHTML = '';
  personalData.innerHTML = '';
  
  // ID Card specific data
  if (jobData.document_type && jobData.document_type.startsWith('id_card')) {
    // Display personal information for ID cards
    if (jobData.personal_info && Object.keys(jobData.personal_info).length > 0) {
      personalData.appendChild(createDataTable(jobData.personal_info));
    } else if (jobData.document_info || jobData.personal_info) {
      // Try to build from document_info and personal_info
      const combinedInfo = {
        ...(jobData.document_info || {}),
        ...(jobData.personal_info || {})
      };
      
      if (Object.keys(combinedInfo).length > 0) {
        personalData.appendChild(createDataTable(combinedInfo));
      } else {
        personalData.innerHTML = '<p class="placeholder">No personal data available</p>';
      }
    } else {
      personalData.innerHTML = '<p class="placeholder">No personal data available</p>';
    }
    
    // Clear vehicle-specific data containers
    vehicleData.innerHTML = '<p class="placeholder">Not applicable for ID cards</p>';
    ownerData.innerHTML = '<p class="placeholder">Not applicable for ID cards</p>';
    registrationData.innerHTML = '<p class="placeholder">Not applicable for ID cards</p>';
  }
  // Vehicle Registration card specific data
  else {
    // Vehicle data
    if (jobData.vehicle_info && Object.keys(jobData.vehicle_info).length > 0) {
      vehicleData.appendChild(createDataTable(jobData.vehicle_info));
    } else if (jobData.vehicle && Object.keys(jobData.vehicle).length > 0) {
      vehicleData.appendChild(createDataTable(jobData.vehicle));
    } else {
      vehicleData.innerHTML = '<p class="placeholder">No vehicle data available</p>';
    }
    
    // Owner data
    if (jobData.owner_info && Object.keys(jobData.owner_info).length > 0) {
      ownerData.appendChild(createDataTable(jobData.owner_info));
    } else if (jobData.owner && Object.keys(jobData.owner).length > 0) {
      ownerData.appendChild(createDataTable(jobData.owner));
    } else {
      ownerData.innerHTML = '<p class="placeholder">No owner data available</p>';
    }
    
    // Registration data
    if (jobData.registration_info && Object.keys(jobData.registration_info).length > 0) {
      registrationData.appendChild(createDataTable(jobData.registration_info));
    } else if (jobData.registration && Object.keys(jobData.registration).length > 0) {
      registrationData.appendChild(createDataTable(jobData.registration));
    } else {
      registrationData.innerHTML = '<p class="placeholder">No registration data available</p>';
    }
    
    // Clear ID card-specific data
    personalData.innerHTML = '<p class="placeholder">Not applicable for vehicle registration</p>';
  }
  
  // Add processing steps information if available
  if (jobData.metadata && jobData.metadata.processing_steps && jobData.metadata.processing_steps.length > 0) {
    const processingInfo = document.createElement('div');
    processingInfo.className = 'processing-info';
    processingInfo.innerHTML = '<h3>Processing Information</h3>';
    
    const stepsList = document.createElement('ul');
    stepsList.className = 'processing-steps';
    
    jobData.metadata.processing_steps.forEach(step => {
      const stepItem = document.createElement('li');
      stepItem.className = step.success ? 'success' : 'error';
      stepItem.innerHTML = `${formatLabel(step.stage)}: <span>${step.success ? '✓' : '✗'}</span>`;
      stepsList.appendChild(stepItem);
    });
    
    processingInfo.appendChild(stepsList);
    
    // Add processing time if available
    if (jobData.metadata.processing_time) {
      const timeInfo = document.createElement('p');
      timeInfo.className = 'processing-time';
      timeInfo.textContent = `Total processing time: ${jobData.metadata.processing_time.toFixed(2)}s`;
      processingInfo.appendChild(timeInfo);
    }
    
    // Add model information if available
    if (jobData.metadata.model_name) {
      const modelInfo = document.createElement('p');
      modelInfo.className = 'model-info';
      modelInfo.textContent = `Model: ${jobData.metadata.model_name}`;
      processingInfo.appendChild(modelInfo);
    }
    
    // Add device information
    if (jobData.metadata.processing_device) {
      const deviceInfo = document.createElement('p');
      deviceInfo.className = 'device-info';
      
      // Check if CPU fallback occurred
      if (jobData.metadata.cpu_fallback) {
        deviceInfo.innerHTML = `Device: <span class="warning">CPU</span> (Fallback due to GPU errors)`;
        deviceInfo.title = `CUDA failures: ${jobData.metadata.cuda_failures || 0}`;
      } else {
        deviceInfo.textContent = `Device: ${jobData.metadata.processing_device.toUpperCase()}`;
      }
      processingInfo.appendChild(deviceInfo);
    }
    
    // Add to appropriate container based on document type
    if (jobData.document_type && jobData.document_type.startsWith('id_card')) {
      personalData.appendChild(processingInfo);
    } else {
      registrationData.appendChild(processingInfo);
    }
  }
  
  // Display extracted JSON (the raw JSON from the VLM)
  if (jobData.extracted_json_str) {
    // Try to pretty-print it if possible
    try {
      const jsonObj = JSON.parse(jobData.extracted_json_str);
      extractedJson.textContent = JSON.stringify(jsonObj, null, 2);
    } catch (e) {
      // If not valid JSON, just display as-is
      extractedJson.textContent = jobData.extracted_json_str;
    }
  } else if (jobData.extracted_json && Object.keys(jobData.extracted_json).length > 0) {
    extractedJson.textContent = JSON.stringify(jobData.extracted_json, null, 2);
  } else {
    extractedJson.textContent = 'No extracted JSON available';
  }
  
  // Display raw data (full API response)
  rawData.textContent = JSON.stringify(jobData, null, 2);
  
  // Enable download button
  document.getElementById('download-btn').disabled = false;
}

/**
 * Create a data table from an object
 * @param {Object} data - Data object to display
 * @returns {HTMLDivElement} - The created table element
 */
function createDataTable(data) {
  const container = document.createElement('div');
  
  for (const [key, value] of Object.entries(data)) {
    // Skip confidence values and objects/arrays
    if (key.endsWith('_confidence') || typeof value === 'object') continue;
    
    const row = document.createElement('div');
    row.className = 'data-row';
    
    const label = document.createElement('div');
    label.className = 'data-label';
    label.textContent = formatLabel(key);
    
    const dataValue = document.createElement('div');
    dataValue.className = 'data-value';
    dataValue.textContent = value;
    
    row.appendChild(label);
    row.appendChild(dataValue);
    container.appendChild(row);
  }
  
  return container;
}

/**
 * Format a label from camelCase or snake_case
 * @param {string} key - The key to format
 * @returns {string} - Formatted label
 */
function formatLabel(key) {
  return key
    .replace(/_/g, ' ')
    .replace(/([A-Z])/g, ' $1')
    .replace(/^./, (str) => str.toUpperCase());
}

/**
 * Set up result tabs
 */
function setupResultTabs() {
  const tabButtons = document.querySelectorAll('.tab-btn');
  const tabContents = document.querySelectorAll('.tab-content');
  
  tabButtons.forEach(button => {
    button.addEventListener('click', () => {
      // Remove active class from all buttons and contents
      tabButtons.forEach(btn => btn.classList.remove('active'));
      tabContents.forEach(content => content.classList.remove('active'));
      
      // Add active class to clicked button and corresponding content
      button.classList.add('active');
      const tabId = button.getAttribute('data-tab');
      document.getElementById(`${tabId}-tab`).classList.add('active');
    });
  });
}

/**
 * Set up action buttons
 */
function setupActionButtons() {
  const downloadBtn = document.getElementById('download-btn');
  const newScanBtn = document.getElementById('new-scan-btn');
  
  if (downloadBtn) {
    downloadBtn.addEventListener('click', () => {
      downloadResults();
    });
  }
  
  if (newScanBtn) {
    newScanBtn.addEventListener('click', () => {
      resetForm();
    });
  }
}

/**
 * Download the results as a JSON file
 */
function downloadResults() {
  const extractedJson = document.getElementById('extracted-json');
  const rawData = document.getElementById('raw-data');
  
  try {
    // Get the selected tab
    const activeTab = document.querySelector('.tab-btn.active').getAttribute('data-tab');
    
    // Based on active tab, decide what to download
    let data, filename;
    
    if (activeTab === 'extracted' && extractedJson && extractedJson.textContent) {
      // Try to parse the extracted JSON
      try {
        data = JSON.parse(extractedJson.textContent);
        filename = 'extracted_json.json';
      } catch (e) {
        // If not valid JSON, download as text
        const blob = new Blob([extractedJson.textContent], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'extracted_data.txt';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        return;
      }
    } else if (rawData && rawData.textContent) {
      // Download full raw data
      data = JSON.parse(rawData.textContent);
      filename = `ocr_results_${data.job_id || 'unknown'}.json`;
    } else {
      alert('No data available to download');
      return;
    }
    
    // Create and download the blob
    const jsonStr = JSON.stringify(data, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  } catch (error) {
    console.error('Error downloading results:', error);
    alert('Failed to download results');
  }
}

/**
 * Reset the form and hide results
 */
function resetForm() {
  // Reset the form
  document.getElementById('upload-form').reset();
  
  // Reset file previews
  document.getElementById('front-preview').style.display = 'none';
  document.getElementById('front-preview').src = '';
  document.getElementById('front-upload').querySelector('.upload-placeholder').style.display = 'flex';
  
  document.getElementById('back-preview').style.display = 'none';
  document.getElementById('back-preview').src = '';
  document.getElementById('back-upload').querySelector('.upload-placeholder').style.display = 'flex';
  
  // Hide results section
  document.getElementById('results-section').style.display = 'none';
}

/**
 * Show the results section
 */
function showResultsSection() {
  const resultsSection = document.getElementById('results-section');
  
  if (!resultsSection) return;
  
  resultsSection.style.display = 'block';
  
  // Scroll to results
  resultsSection.scrollIntoView({ behavior: 'smooth' });
}

/**
 * Simulate job progress for development purposes
 * @param {string} jobId - Job ID
 */
function simulateJobProgress(jobId) {
  let progress = 0;
  const stages = [
    { progress: 10, status: 'queued', message: 'Job queued, waiting to start...' },
    { progress: 25, status: 'processing', message: 'Analyzing front side of card...' },
    { progress: 50, status: 'processing', message: 'Analyzing back side of card...' },
    { progress: 75, status: 'processing', message: 'Extracting structured data...' },
    { progress: 90, status: 'processing', message: 'Validating results...' },
    { progress: 100, status: 'completed', message: 'Processing completed successfully' }
  ];
  
  let stageIndex = 0;
  
  // Update status right away for the first stage
  updateStatusIndicator('loading', stages[0].message);
  updateProgressBar(stages[0].progress);
  
  const interval = setInterval(() => {
    stageIndex++;
    
    if (stageIndex >= stages.length) {
      clearInterval(interval);
      
      // Display mock results
      const mockResult = getMockResults(jobId);
      displayResults(mockResult);
      
      // Final status update
      updateStatusIndicator('success', 'Processing completed successfully');
      updateProgressBar(100);
      return;
    }
    
    const stage = stages[stageIndex];
    updateStatusIndicator('loading', stage.message);
    updateProgressBar(stage.progress);
  }, 1500);
}

/**
 * Get mock results for development
 * @param {string} jobId - Job ID
 * @returns {Object} - Mock result data
 */
function getMockResults(jobId) {
  return {
    job_id: jobId,
    status: 'completed',
    created_at: new Date(Date.now() - 5000).toISOString(),
    completed_at: new Date().toISOString(),
    vehicle: {
      make: 'Toyota',
      model: 'Camry',
      year: '2020',
      vin: '1HGBH41JXMN109186',
      color: 'Silver',
      type: 'Sedan',
      engine: '2.5L 4-Cylinder',
      weight: '3,310 lbs',
      make_confidence: 0.98,
      model_confidence: 0.96
    },
    owner: {
      name: 'John Smith',
      address: '123 Main St, Anytown, CA 12345',
      license_number: 'D12345678',
      date_of_birth: '1985-06-15',
      name_confidence: 0.95
    },
    registration: {
      registration_number: 'REG-12345-678',
      issue_date: '2022-01-15',
      expiration_date: '2023-01-15',
      fee: '$120.00',
      status: 'Valid',
      state: 'California',
      plate_number: 'ABC-1234'
    },
    additional_fields: {
      insurance_policy: 'POL-987654',
      insurance_company: 'Reliable Insurance Co.',
      emissions_date: '2021-12-10'
    },
    metadata: {
      processing_time: 3.45,
      model_version: 'Qwen/Qwen2.5-VL-7B-Instruct-AWQ',
      front_image_processed: true,
      back_image_processed: true,
      confidence_score: 0.92
    },
    validation: {
      overall_quality: 'excellent',
      valid_fields: 18,
      invalid_fields: 0,
      warnings: []
    }
  };
}

/**
 * Fetch just the extracted JSON from the API
 * @param {string} jobId - Job ID
 * @returns {Promise<Object>} The parsed JSON data
 */
async function fetchExtractedJson(jobId) {
  const extractedJson = document.getElementById('extracted-json');
  if (!extractedJson) return null;
  
  try {
    console.log(`Fetching JSON data for job ${jobId}...`);
    const response = await fetch(`${API_CONFIG.baseUrl}${API_CONFIG.endpoints.extractedJson}${jobId}/json`);
    
    if (!response.ok) {
      console.warn('Could not fetch extracted JSON directly');
      return null;
    }
    
    const data = await response.json();
    let resultData = null;
    
    console.log("Raw API response:", data);
    
    // If we got parsed JSON data, display it
    if (data && !data.raw_json_string && !data.raw_response) {
      extractedJson.textContent = JSON.stringify(data, null, 2);
      resultData = data;
      console.log("Using direct JSON data");
    } 
    // If we got a raw JSON string, display it and try to parse it
    else if (data && data.raw_json_string) {
      extractedJson.textContent = data.raw_json_string;
      try {
        resultData = JSON.parse(data.raw_json_string);
        console.log("Parsed from raw_json_string");
      } catch (e) {
        console.error('Error parsing raw JSON string:', e);
      }
    }
    // If we got raw response, display it as-is
    else if (data && data.raw_response) {
      extractedJson.textContent = data.raw_response;
      try {
        resultData = JSON.parse(data.raw_response);
        console.log("Parsed from raw_response");
      } catch (e) {
        console.error('Error parsing raw response:', e);
      }
    }
    
    // If we successfully got data, try to display it right away
    if (resultData && window.OCRApp && window.OCRApp.displayResults) {
      console.log("Immediately displaying with OCRApp.displayResults");
      window.OCRApp.displayResults(resultData);
    }
    
    return resultData;
  } catch (error) {
    console.error('Error fetching extracted JSON:', error);
    return null;
  }
}

/**
 * Handle and display face detection results
 * @param {Object} faceData - Face detection data
 */
function displayFaceDetectionResults(faceData) {
  const faceContainer = document.getElementById('face-detection-data');
  
  if (!faceContainer || !faceData) {
    return;
  }
  
  // Clear existing content
  faceContainer.innerHTML = '';
  
  // Create face detection container
  const container = document.createElement('div');
  container.className = 'face-detection-container';
  
  // Create header
  const header = document.createElement('div');
  header.className = 'face-detection-header';
  
  const title = document.createElement('h3');
  title.textContent = 'Face Detection Results';
  
  const status = document.createElement('span');
  status.className = 'face-detection-status';
  
  if (faceData.detection_status) {
    status.textContent = 'DETECTED';
    status.classList.add('detected');
  } else {
    status.textContent = 'NOT DETECTED';
    status.classList.add('not-detected');
  }
  
  header.appendChild(title);
  header.appendChild(status);
  container.appendChild(header);
  
  // Add detection details
  if (faceData.detection_status && faceData.detections && faceData.detections.length > 0) {
    const detectionInfo = document.createElement('div');
    detectionInfo.className = 'detection-info';
    
    const infoList = document.createElement('ul');
    
    // Add basic detection info
    infoList.innerHTML = `
      <li><strong>Number of detections:</strong> ${faceData.num_detections || 0}</li>
      <li><strong>Detection time:</strong> ${(faceData.detection_time || 0).toFixed(3)}s</li>
      <li><strong>Feature extraction time:</strong> ${(faceData.feature_extraction_time || 0).toFixed(3)}s</li>
    `;
    
    detectionInfo.appendChild(infoList);
    container.appendChild(detectionInfo);
    
    // Add embeddings section if available
    if (faceData.embeddings && Object.keys(faceData.embeddings).length > 0) {
      const embeddingsContainer = document.createElement('div');
      embeddingsContainer.className = 'embeddings-container';
      
      const embeddingsHeader = document.createElement('div');
      embeddingsHeader.className = 'embeddings-header';
      embeddingsHeader.textContent = 'ResNet-50 Embeddings';
      
      const toggleButton = document.createElement('button');
      toggleButton.className = 'embeddings-toggle';
      toggleButton.textContent = 'Show Embeddings';
      
      const embeddingsData = document.createElement('pre');
      embeddingsData.className = 'embeddings-data';
      
      // Format embeddings info
      const embeddingsInfo = {
        version: faceData.embeddings.version,
        shape: faceData.embeddings.shape,
        extraction_time: faceData.embeddings.extraction_time,
        data_preview: Array.isArray(faceData.embeddings.data) ? 
                     `${faceData.embeddings.data.slice(0, 10).map(n => n.toFixed(4))}... (${faceData.embeddings.data.length} values)` : 
                     'No data'
      };
      
      embeddingsData.textContent = JSON.stringify(embeddingsInfo, null, 2);
      
      // Set up toggle button
      toggleButton.addEventListener('click', function() {
        embeddingsData.classList.toggle('active');
        toggleButton.textContent = embeddingsData.classList.contains('active') ? 
                                  'Hide Embeddings' : 'Show Embeddings';
      });
      
      embeddingsContainer.appendChild(embeddingsHeader);
      embeddingsContainer.appendChild(toggleButton);
      embeddingsContainer.appendChild(embeddingsData);
      container.appendChild(embeddingsContainer);
    }
  } else {
    // No detection info
    const noDetection = document.createElement('p');
    noDetection.textContent = 'No face or person detected in this document.';
    container.appendChild(noDetection);
  }
  
  faceContainer.appendChild(container);
} 