document.addEventListener('DOMContentLoaded', () => {
  // Mobile Navigation Toggle
  const hamburger = document.querySelector('.hamburger');
  const navMenu = document.querySelector('.nav-menu');
  
  if (hamburger) {
    hamburger.addEventListener('click', () => {
      hamburger.classList.toggle('active');
      navMenu.classList.toggle('active');
    });
  }
  
  // Close menu when clicking nav links (mobile)
  const navLinks = document.querySelectorAll('.nav-link');
  navLinks.forEach(link => {
    link.addEventListener('click', () => {
      if (hamburger && navMenu) {
        hamburger.classList.remove('active');
        navMenu.classList.remove('active');
      }
    });
  });

  // Data type card selection
  const dataCards = document.querySelectorAll('.data-card');
  if (dataCards.length > 0) {
    dataCards.forEach(card => {
      card.addEventListener('click', () => {
        // Remove selected class from all cards
        dataCards.forEach(c => c.classList.remove('selected'));
        // Add selected class to clicked card
        card.classList.add('selected');
        
        // Show relevant configuration section based on data type
        const dataType = card.getAttribute('data-type');
        showConfigSection(dataType);
      });
    });
  }

  // Function to show relevant configuration section
  function showConfigSection(dataType) {
    // Hide all config sections
    const configSections = document.querySelectorAll('.config-section');
    configSections.forEach(section => {
      section.style.display = 'none';
    });
    
    // Show the relevant section
    const relevantSection = document.getElementById(`${dataType}-config`);
    if (relevantSection) {
      relevantSection.style.display = 'block';
    }
  }

  // Form submission with loading indicators and AJAX request
  const forgeForm = document.getElementById('forge-form');
  
  if (forgeForm) {
    // Ensure all submit buttons within the form trigger with the right data type
    const generateButtons = forgeForm.querySelectorAll('button[type="submit"]');
    
    generateButtons.forEach(button => {
      button.addEventListener('click', function(e) {
        e.preventDefault();
        
        // Get the data type from button id or parent section
        let dataType = '';
        const configSection = this.closest('.config-section');
        if (configSection) {
          dataType = configSection.id.replace('-config', '');
        }
        
        // Show loading state
        const submitButton = this;
        const originalButtonText = submitButton.textContent;
        submitButton.innerHTML = 'Generating <span class="spinner"></span>';
        submitButton.disabled = true;
        
        // Collect form data specific to this config section
        const formData = {};
        const inputs = configSection.querySelectorAll('input, select, textarea');
        
        inputs.forEach(input => {
          if (input.type === 'checkbox') {
            if (input.checked) {
              // Handle checkbox groups
              if (input.name.includes('components')) {
                if (!formData[input.name]) {
                  formData[input.name] = [];
                }
                formData[input.name].push(input.value);
              } else {
                formData[input.name] = input.value;
              }
            }
          } else if (input.type === 'radio') {
            if (input.checked) {
              formData[input.name] = input.value;
            }
          } else {
            formData[input.name] = input.value;
          }
        });
        
        // Convert string numbers to actual numbers
        for (const key in formData) {
          if (formData[key] !== '' && !isNaN(formData[key])) {
            formData[key] = Number(formData[key]);
          }
        }
        
        // Make AJAX request to the server
        fetch('/api/generate', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            data_type: dataType,
            config: formData
          })
        })
        .then(response => response.json())
        .then(data => {
          // Hide loading state
          submitButton.innerHTML = originalButtonText;
          submitButton.disabled = false;
          
          if (data.success) {
            // Create file info based on data type
            let fileInfoHtml = '';
            
            if (dataType === 'tabular') {
              fileInfoHtml = `
                <div class="file-info">
                  <div><strong>Rows:</strong> ${data.file_info.rows}</div>
                  <div><strong>Columns:</strong> ${data.file_info.columns}</div>
                  <div><strong>Format:</strong> ${data.file_info.format}</div>
                  <div><strong>File:</strong> ${data.file_info.file_name}</div>
                </div>
              `;
            } else if (dataType === 'image') {
              fileInfoHtml = `
                <div class="file-info">
                  <div><strong>Width:</strong> ${data.file_info.width}px</div>
                  <div><strong>Height:</strong> ${data.file_info.height}px</div>
                  <div><strong>Images:</strong> ${data.file_info.num_images}</div>
                  <div><strong>Format:</strong> ${data.file_info.format}</div>
                </div>
              `;
              
              // If it's a single image, show a preview
              if (data.file_info.num_images === 1 && 
                  ['png', 'jpg', 'jpeg', 'gif', 'webp'].includes(data.file_info.format.toLowerCase())) {
                // Use the view_url if provided, otherwise use the download_url
                const imageUrl = data.file_info.view_url || data.file_info.download_url;
                fileInfoHtml += `
                  <div class="image-preview">
                    <img src="${imageUrl}" alt="Generated Image" style="max-width: 100%; max-height: 300px; margin: 20px 0; border: 1px solid #ddd; border-radius: 5px;">
                  </div>
                `;
              }
            } else {
              fileInfoHtml = `
                <div class="file-info">
                  <div><strong>Format:</strong> ${data.file_info.format}</div>
                  <div><strong>File:</strong> ${data.file_info.file_name}</div>
                </div>
              `;
            }
            
            // Show success message with download link
            configSection.innerHTML = `
              <div class="success-message">
                <h3><i class="fas fa-check-circle"></i> Data Generated Successfully!</h3>
                <p>Your synthetic ${dataType} data has been created with the following properties:</p>
                ${fileInfoHtml}
                <div class="download-container">
                  <a href="${data.file_info.download_url}" class="btn btn-primary">Download Data</a>
                  <button class="btn btn-outline" id="generate-more">Generate More Data</button>
                </div>
              </div>
            `;
            
            // Handle "Generate More" button
            document.getElementById('generate-more').addEventListener('click', function() {
              window.location.reload();
            });
          } else {
            // Show error message
            alert('Error generating data: ' + data.message);
          }
        })
        .catch(error => {
          // Hide loading state
          submitButton.innerHTML = originalButtonText;
          submitButton.disabled = false;
          
          // Show error message
          alert('Error: ' + error.message);
          console.error('Error:', error);
        });
      });
    });
  }

  // Add CSS for file info display
  const style = document.createElement('style');
  
  // Add CSS for file info display
  style.textContent = `
    .file-info {
      background-color: #f5f5f5;
      border-radius: 5px;
      padding: 15px;
      margin: 20px 0;
      text-align: left;
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 10px;
    }
    
    .download-container {
      margin-top: 25px;
      display: flex;
      justify-content: center;
      gap: 15px;
    }
  `;
  
  // Append the style element to the head
  document.head.appendChild(style);

  // Hero section animated effect for non-particle version
  const hero = document.querySelector('.hero');
  if (hero) {
    window.addEventListener('scroll', () => {
      const scrollPosition = window.scrollY;
      if (scrollPosition < window.innerHeight) {
        // Parallax effect
        hero.style.backgroundPosition = `center ${scrollPosition * 0.5}px`;
      }
    });
  }
  
  // Schema type handling for tabular data
  const schemaType = document.getElementById('schema-type');
  const customSchemaContainer = document.getElementById('custom-schema-container');
  
  if (schemaType && customSchemaContainer) {
    schemaType.addEventListener('change', function() {
      if (this.value === 'custom') {
        customSchemaContainer.style.display = 'block';
      } else {
        customSchemaContainer.style.display = 'none';
      }
    });
  }
  
  // Add animation classes to elements when they come into view
  const animateOnScroll = function() {
    const animatedElements = document.querySelectorAll('.animate-on-scroll');
    
    animatedElements.forEach(element => {
      const rect = element.getBoundingClientRect();
      const windowHeight = window.innerHeight || document.documentElement.clientHeight;
      
      if (rect.top <= windowHeight * 0.9) {
        const animationClass = element.dataset.animation || 'animate__fadeIn';
        element.classList.add('animate__animated', animationClass);
      }
    });
  };
  
  // Run animation check on load and scroll
  if (document.querySelectorAll('.animate-on-scroll').length > 0) {
    animateOnScroll();
    window.addEventListener('scroll', animateOnScroll);
  }
}); 

function handleFormSubmit(event) {
    event.preventDefault();
    
    // Show loading animation
    const configSection = document.getElementById('configSection');
    configSection.innerHTML = `
        <div class="loading-container">
            <div class="loading-animation">
                <div class="forge-animation">
                    <div class="hammer"></div>
                    <div class="anvil"></div>
                    <div class="sparks"></div>
                </div>
            </div>
            <p>Forging your synthetic data...</p>
        </div>
    `;
    
    // Get form data
    const formData = new FormData(event.target);
    const dataType = formData.get('dataType');
    
    // Prepare request data
    let requestData = {
        data_type: dataType
    };
    
    // Add type-specific parameters
    if (dataType === 'tabular') {
        requestData.num_rows = parseInt(formData.get('numRows'), 10) || 100;
        requestData.num_cols = parseInt(formData.get('numCols'), 10) || 5;
        requestData.format = formData.get('format') || 'csv';
        
        // Add schema if provided
        const schemaInput = document.getElementById('schemaInput');
        if (schemaInput && schemaInput.value.trim()) {
            try {
                requestData.schema = JSON.parse(schemaInput.value);
            } catch (e) {
                console.error('Invalid schema JSON:', e);
            }
        }
    } else if (dataType === 'image') {
        requestData.image_type = formData.get('imageType') || 'noise';
        requestData.width = parseInt(formData.get('width'), 10) || 512;
        requestData.height = parseInt(formData.get('height'), 10) || 512;
        requestData.num_images = parseInt(formData.get('numImages'), 10) || 1;
        requestData.format = formData.get('imageFormat') || 'png';
        requestData.color_mode = formData.get('colorMode') || 'RGB';
    }
    
    // Send AJAX request
    fetch('/api/generate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Create file info HTML based on data type
            let fileInfoHtml = '';
            
            if (dataType === 'tabular') {
                // Safely access properties with default values if they don't exist
                const rows = data.rows || 0;
                const columns = data.columns || 0;
                const format = data.format || 'csv';
                const filename = data.filename || 'data.csv';
                
                fileInfoHtml = `
                    <div class="file-info">
                        <p><strong>Rows:</strong> ${rows}</p>
                        <p><strong>Columns:</strong> ${columns}</p>
                        <p><strong>Format:</strong> ${format}</p>
                        <p><strong>Filename:</strong> ${filename}</p>
                    </div>
                `;
            } else if (dataType === 'image') {
                // Safely access properties with default values if they don't exist
                const width = data.width || 512;
                const height = data.height || 512;
                const num_images = data.num_images || 1;
                const format = data.format || 'png';
                const filename = data.filename || 'image.png';
                
                fileInfoHtml = `
                    <div class="file-info">
                        <p><strong>Width:</strong> ${width}px</p>
                        <p><strong>Height:</strong> ${height}px</p>
                        <p><strong>Images:</strong> ${num_images}</p>
                        <p><strong>Format:</strong> ${format}</p>
                        <p><strong>Filename:</strong> ${filename}</p>
                    </div>
                `;
                
                // Add image preview if available
                if (data.view_url) {
                    fileInfoHtml += `
                        <div class="image-preview">
                            <h4>Preview:</h4>
                            <img src="${data.view_url}" alt="Generated image" style="max-width: 100%; max-height: 300px;">
                        </div>
                    `;
                }
            }
            
            // Display success message with download link and file info
            configSection.innerHTML = `
                <div class="success-message">
                    <div class="success-icon">✓</div>
                    <h3>Data Successfully Forged!</h3>
                    <p>${data.message || 'Your data has been generated successfully!'}</p>
                    ${fileInfoHtml}
                    <div class="download-container">
                        <a href="${data.download_url || '#'}" class="download-button">
                            <span class="download-icon">↓</span>
                            Download File
                        </a>
                    </div>
                    <button onclick="location.reload()" class="reload-button">Create Another</button>
                </div>
            `;
        } else {
            // Display error message
            configSection.innerHTML = `
                <div class="error-message">
                    <div class="error-icon">⚠</div>
                    <h3>Error Generating Data</h3>
                    <p>${data.message || 'An error occurred while generating the data.'}</p>
                    <button onclick="location.reload()" class="reload-button">Try Again</button>
                </div>
            `;
        }
    })
    .catch(error => {
        console.error('Error:', error);
        configSection.innerHTML = `
            <div class="error-message">
                <div class="error-icon">⚠</div>
                <h3>Error Generating Data</h3>
                <p>An error occurred while generating the data. Please try again.</p>
                <button onclick="location.reload()" class="reload-button">Try Again</button>
            </div>
        `;
    });
} 