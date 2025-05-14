// Bundled JavaScript file for the Vehicle Registration Card OCR application
// This file is generated from the source files and should not be edited directly

// Create a global namespace for our functions
window.OCRApp = window.OCRApp || {};

// Main application bundle
(function() {
  // Helper functions
  function $(selector) {
    return document.querySelector(selector);
  }
  
  function $$(selector) {
    return document.querySelectorAll(selector);
  }

  // Initialize the application when the DOM is loaded
  document.addEventListener('DOMContentLoaded', function() {
    initApp();
  });

  function initApp() {
    // Check which page we're on
    const isHomePage = window.location.pathname === '/' || 
                      window.location.pathname.includes('index.html');
    const isUploadPage = window.location.pathname.includes('upload.html');
    
    if (isHomePage) {
      initHomePage();
    } else if (isUploadPage) {
      initUploadPage();
    }
  }

  function initHomePage() {
    // Setup demo button
    const demoButton = $('.secondary-btn');
    if (demoButton) {
      demoButton.addEventListener('click', function() {
        alert('Demo functionality coming soon!');
      });
    }
  }

  function initUploadPage() {
    // Wait for the upload.js to load and initialize
    console.log('Upload page initialized by index-BuKCFbWJ.js');
    
    // Set up tab navigation
    setupTabNavigation();
    
    // Try to load sample data if we're on the results page
    const resultsSection = $('#results-section');
    if (resultsSection && resultsSection.style.display !== 'none') {
      loadSampleData();
    }
  }
  
  // Set up tab navigation
  function setupTabNavigation() {
    const tabButtons = $$('.tab-btn');
    const tabContents = $$('.tab-content');
    
    tabButtons.forEach(button => {
      button.addEventListener('click', () => {
        // Remove active class from all buttons and contents
        tabButtons.forEach(btn => btn.classList.remove('active'));
        tabContents.forEach(content => content.classList.remove('active'));
        
        // Add active class to clicked button
        button.classList.add('active');
        
        // Show corresponding content
        const tabName = button.getAttribute('data-tab');
        $('#' + tabName + '-tab').classList.add('active');
      });
    });
  }
  
  // Load sample data directly from the user's example
  function loadSampleData() {
    const sampleData = {
      "vehicle_info": {
        "make": " هيونداي",
        "car_model": "STARIA",
        "body_type": "فان للركاب",
        "plate_type": "نوع اللوحة نقل خاص",
        "plate_number": "228119",
        "year_of_manufacture": "2024",
        "country_of_manufacture": "كوريا الجنوبية",
        "cylinders": "06",
        "seats": "011",
        "chassis_number": "KMJYA3745RU137428",
        "engine_number": "G6DUPA225171",
        "first_registration_date": "2023-10-08"
      },
      "owner_info": {
        "name": "IBRAHIM MOHAMMED A. A. ALMOHANNADI",
        "id": "27363401041",
        "nationality": "قطرى"
      },
      "registration_info": {
        "registration_date": "2023-10-08",
        "expiry_date": "2024-10-07",
        "renew_date": "2023-10-08"
      },
      "insurance_info": {
        "insurance_company": "مجموعة الدوحة للتأمين",
        "policy_number": "P/161212/2023/P",
        "expiry_date": "2024-10-07"
      },
      "metadata": {
        "process_id": "ocr_ac10709c4adf44cc9d303ec42009acfd",
        "processing_time": 7.300262928009033,
        "model_id": "unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit",
        "model_name": "unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit"
      },
      "raw_model_response": ""
    };
    
    console.log("Loading sample data directly from example");
    
    // Display the data in the UI
    displayResults(sampleData);
    
    // Also update the extracted JSON display
    const extractedJson = $('#extracted-json');
    if (extractedJson) {
      extractedJson.textContent = JSON.stringify(sampleData, null, 2);
    }
  }
  
  // Display the results in the appropriate tabs
  function displayResults(data) {
    console.log("Displaying results from OCR data:", data);
    
    // Vehicle Information
    if (data.vehicle_info) {
      displayVehicleInfo(data.vehicle_info);
    }
    
    // Owner Information
    if (data.owner_info) {
      displayOwnerInfo(data.owner_info);
    }
    
    // Registration Information
    if (data.registration_info || data.insurance_info) {
      displayRegistrationInfo(data.registration_info, data.insurance_info);
    }
  }
  
  // Format and display vehicle information
  function displayVehicleInfo(vehicleInfo) {
    const vehicleDataElement = $('#vehicle-data');
    if (!vehicleDataElement) return;
    
    console.log("Displaying vehicle info:", vehicleInfo);
    
    let html = '<div class="data-card">';
    
    // Vehicle main details
    html += '<div class="data-section">';
    html += '<h3 class="section-title">Vehicle Details</h3>';
    html += '<div class="info-grid">';
    
    // Add each vehicle detail with proper formatting
    if (vehicleInfo.make) html += createInfoItem('Make', vehicleInfo.make);
    if (vehicleInfo.car_model) html += createInfoItem('Model', vehicleInfo.car_model);
    if (vehicleInfo.year_of_manufacture) html += createInfoItem('Year', vehicleInfo.year_of_manufacture);
    if (vehicleInfo.body_type) html += createInfoItem('Body Type', vehicleInfo.body_type);
    if (vehicleInfo.country_of_manufacture) html += createInfoItem('Origin', vehicleInfo.country_of_manufacture);
    
    html += '</div></div>';
    
    // Vehicle registration details
    html += '<div class="data-section">';
    html += '<h3 class="section-title">Vehicle Registration</h3>';
    html += '<div class="info-grid">';
    
    if (vehicleInfo.plate_number) html += createInfoItem('Plate Number', vehicleInfo.plate_number);
    if (vehicleInfo.plate_type) html += createInfoItem('Plate Type', vehicleInfo.plate_type);
    if (vehicleInfo.first_registration_date) html += createInfoItem('First Registration', vehicleInfo.first_registration_date);
    
    html += '</div></div>';
    
    // Technical specifications
    html += '<div class="data-section">';
    html += '<h3 class="section-title">Technical Specifications</h3>';
    html += '<div class="info-grid">';
    
    if (vehicleInfo.chassis_number) html += createInfoItem('Chassis Number', vehicleInfo.chassis_number);
    if (vehicleInfo.engine_number) html += createInfoItem('Engine Number', vehicleInfo.engine_number);
    if (vehicleInfo.cylinders) html += createInfoItem('Cylinders', vehicleInfo.cylinders);
    if (vehicleInfo.seats) html += createInfoItem('Seats', vehicleInfo.seats);
    
    html += '</div></div>';
    html += '</div>'; // Close data-card
    
    vehicleDataElement.innerHTML = html;
  }
  
  // Format and display owner information
  function displayOwnerInfo(ownerInfo) {
    const ownerDataElement = $('#owner-data');
    if (!ownerDataElement) return;
    
    console.log("Displaying owner info:", ownerInfo);
    
    let html = '<div class="data-card">';
    
    // Owner details
    html += '<div class="data-section">';
    html += '<h3 class="section-title">Owner Details</h3>';
    html += '<div class="info-grid">';
    
    if (ownerInfo.name) html += createInfoItem('Name', ownerInfo.name);
    if (ownerInfo.id) html += createInfoItem('ID Number', ownerInfo.id);
    if (ownerInfo.nationality) html += createInfoItem('Nationality', ownerInfo.nationality);
    
    html += '</div></div>';
    html += '</div>'; // Close data-card
    
    ownerDataElement.innerHTML = html;
  }
  
  // Format and display registration information
  function displayRegistrationInfo(registrationInfo, insuranceInfo) {
    const registrationDataElement = $('#registration-data');
    if (!registrationDataElement) return;
    
    console.log("Displaying registration info:", registrationInfo, insuranceInfo);
    
    let html = '<div class="data-card">';
    
    // Registration details
    if (registrationInfo) {
      html += '<div class="data-section">';
      html += '<h3 class="section-title">Registration Details</h3>';
      html += '<div class="info-grid">';
      
      if (registrationInfo.registration_date) html += createInfoItem('Registration Date', registrationInfo.registration_date);
      if (registrationInfo.expiry_date) html += createInfoItem('Expiry Date', registrationInfo.expiry_date);
      if (registrationInfo.renew_date) html += createInfoItem('Renewal Date', registrationInfo.renew_date);
      
      html += '</div></div>';
    }
    
    // Insurance details
    if (insuranceInfo) {
      html += '<div class="data-section">';
      html += '<h3 class="section-title">Insurance Details</h3>';
      html += '<div class="info-grid">';
      
      if (insuranceInfo.insurance_company) html += createInfoItem('Insurance Company', insuranceInfo.insurance_company);
      if (insuranceInfo.policy_number) html += createInfoItem('Policy Number', insuranceInfo.policy_number);
      if (insuranceInfo.expiry_date) html += createInfoItem('Expiry Date', insuranceInfo.expiry_date);
      
      html += '</div></div>';
    }
    
    html += '</div>'; // Close data-card
    
    registrationDataElement.innerHTML = html;
  }
  
  // Helper function to create an info item
  function createInfoItem(label, value) {
    return `
      <div class="info-item">
        <div class="info-label">${label}</div>
        <div class="info-value">${value}</div>
      </div>
    `;
  }
  
  // Load data directly from text
  window.OCRApp.loadDataFromText = function(jsonText) {
    try {
      const data = JSON.parse(jsonText);
      console.log("Parsed data from text:", data);
      displayResults(data);
      return true;
    } catch (e) {
      console.error("Error parsing JSON text:", e);
      return false;
    }
  };
  
  // Expose public functions
  window.OCRApp.displayResults = displayResults;
  window.OCRApp.displayVehicleInfo = displayVehicleInfo;
  window.OCRApp.displayOwnerInfo = displayOwnerInfo;
  window.OCRApp.displayRegistrationInfo = displayRegistrationInfo;
  window.OCRApp.loadSampleData = loadSampleData;
})(); 