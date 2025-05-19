/**
 * Main JavaScript for VehicleScan AI - OCR Application
 */

// API configuration
const API_CONFIG = {
  baseUrl: window.location.hostname === 'localhost' ? 'http://localhost:8000' : 'http://'+window.location.hostname+':8000',
  endpoints: {
    process: '/api/process',
    jobStatus: '/api/jobs/'
  }
};

document.addEventListener('DOMContentLoaded', () => {
  // Initialize the application
  initApp();
  
  // Log API base URL
  console.log('API base URL:', API_CONFIG.baseUrl);
});

/**
 * Initialize the application
 */
function initApp() {
  // Set up event listeners
  setupEventListeners();
  
  // Display welcome message in console
  console.log('VehicleScan AI Frontend initialized');
  
  // Add some animation to the hero section
  animateHero();
}

/**
 * Set up event listeners for interactive elements
 */
function setupEventListeners() {
  // Get UI elements
  const startScanBtn = document.querySelector('.primary-btn');
  const demoBtn = document.querySelector('.secondary-btn');
  const navLinks = document.querySelectorAll('nav a');
  
  // Start scan button is now a link to upload.html
  
  // Demo button
  if (demoBtn) {
    demoBtn.addEventListener('click', () => {
      // This would normally show a demo
      alert('Demo mode activated! This will demonstrate the OCR capabilities.');
    });
  }
  
  // Navigation links
  navLinks.forEach(link => {
    // Skip if it's a direct page link
    if (link.getAttribute('href').endsWith('.html')) {
      return;
    }
    
    link.addEventListener('click', (e) => {
      // Prevent default behavior
      e.preventDefault();
      
      // Remove active class from all links
      navLinks.forEach(l => l.classList.remove('active'));
      
      // Add active class to clicked link
      link.classList.add('active');
      
      // Get the section name from the href
      const sectionName = link.textContent.toLowerCase();
      
      // Show message for navigation
      console.log(`Navigating to ${sectionName} section`);
      alert(`You clicked on the ${sectionName} section. This would navigate to the ${sectionName} page.`);
    });
  });
}

/**
 * Add animations to the hero section
 */
function animateHero() {
  const cardOutline = document.querySelector('.card-outline');
  
  if (cardOutline) {
    // Add a subtle animation effect
    setTimeout(() => {
      cardOutline.style.transition = 'all 0.8s ease-in-out';
      cardOutline.style.transform = 'scale(1.02)';
      cardOutline.style.boxShadow = '0 6px 12px rgba(0, 0, 0, 0.1)';
      
      // Reset after animation
      setTimeout(() => {
        cardOutline.style.transform = 'scale(1)';
        cardOutline.style.boxShadow = 'none';
      }, 1000);
    }, 1000);
  }
}

/**
 * Mock API service for OCR functionality
 * In a real application, this would connect to the backend
 */
class OCRService {
  /**
   * Upload an image for OCR processing
   * @param {File} file - The image file to process
   * @returns {Promise} - Promise that resolves to the job ID
   */
  static uploadImage(file) {
    // For real API calls
    if (API_CONFIG && API_CONFIG.baseUrl) {
      const formData = new FormData();
      formData.append('front', file);
      
      return fetch(`${API_CONFIG.baseUrl}${API_CONFIG.endpoints.process}`, {
        method: 'POST',
        body: formData
      })
      .then(response => {
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        return response.json();
      })
      .catch(error => {
        console.error('Error uploading image:', error);
        // Fall back to simulation
        return this.simulateUpload(file);
      });
    }
    
    // Simulation as fallback
    return this.simulateUpload(file);
  }
  
  /**
   * Simulate an API upload response
   * @param {File} file - The image file
   * @returns {Promise} - Simulated response
   */
  static simulateUpload(file) {
    return new Promise((resolve) => {
      console.log(`Processing file: ${file.name}`);
      // Simulate API call
      setTimeout(() => {
        resolve({
          job_id: 'ocr_' + Math.random().toString(36).substr(2, 9),
          status: 'queued'
        });
      }, 1500);
    });
  }
  
  /**
   * Get the status and results of an OCR job
   * @param {string} jobId - The job ID to check
   * @returns {Promise} - Promise that resolves to the job status and results
   */
  static getJobStatus(jobId) {
    // For real API calls
    if (API_CONFIG && API_CONFIG.baseUrl) {
      return fetch(`${API_CONFIG.baseUrl}${API_CONFIG.endpoints.jobStatus}${jobId}`)
        .then(response => {
          if (!response.ok) {
            throw new Error('Network response was not ok');
          }
          return response.json();
        })
        .catch(error => {
          console.error('Error fetching job status:', error);
          // Fall back to simulation
          return this.simulateJobStatus(jobId);
        });
    }
    
    // Simulation as fallback
    return this.simulateJobStatus(jobId);
  }
  
  /**
   * Simulate a job status response
   * @param {string} jobId - The job ID
   * @returns {Promise} - Simulated response
   */
  static simulateJobStatus(jobId) {
    return new Promise((resolve) => {
      // Simulate API call
      setTimeout(() => {
        resolve({
          job_id: jobId,
          status: 'completed',
          vehicle: {
            make: 'Toyota',
            model: 'Camry',
            year: '2020',
            vin: '1HGBH41JXMN109186'
          },
          owner: {
            name: 'John Smith',
            address: '123 Main St, Anytown, CA 12345'
          }
        });
      }, 2000);
    });
  }
}

// Export for module usage if needed
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { OCRService };
} 