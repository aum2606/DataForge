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

  // Form validation
  const forgeForm = document.getElementById('forge-form');
  if (forgeForm) {
    forgeForm.addEventListener('submit', (event) => {
      event.preventDefault();
      
      // Validate form inputs
      const isValid = validateForm();
      
      if (isValid) {
        // Show loading animation
        const forgeButton = document.querySelector('#forge-form button[type="submit"]');
        if (forgeButton) {
          const originalText = forgeButton.innerHTML;
          forgeButton.innerHTML = 'Forging... <span class="spinner"></span>';
          forgeButton.disabled = true;
          
          // Simulate data generation (to be replaced with actual API call)
          setTimeout(() => {
            forgeButton.innerHTML = originalText;
            forgeButton.disabled = false;
            
            // Show success message
            const resultSection = document.getElementById('result-section');
            if (resultSection) {
              resultSection.innerHTML = `
                <div class="success-message animate__animated animate__fadeIn">
                  <h3>Data Generated Successfully!</h3>
                  <p>Your synthetic data has been forged and is ready for download.</p>
                  <a href="#" class="btn btn-primary forge-btn-animated">Download Data</a>
                </div>
              `;
              resultSection.style.display = 'block';
              
              // Smooth scroll to result section
              resultSection.scrollIntoView({ behavior: 'smooth' });
            }
          }, 2000);
        }
      }
    });
  }
  
  // Form validation function
  function validateForm() {
    let isValid = true;
    const inputs = document.querySelectorAll('.form-control[required]');
    
    inputs.forEach(input => {
      if (!input.value.trim()) {
        isValid = false;
        input.classList.add('error');
        
        // Add error message
        const errorElement = document.createElement('div');
        errorElement.className = 'error-message';
        errorElement.textContent = 'This field is required';
        
        // Only add error message if it doesn't exist
        if (!input.parentElement.querySelector('.error-message')) {
          input.parentElement.appendChild(errorElement);
        }
      } else {
        input.classList.remove('error');
        const errorElement = input.parentElement.querySelector('.error-message');
        if (errorElement) {
          errorElement.remove();
        }
      }
    });
    
    return isValid;
  }

  // Add input event listeners to clear errors on typing
  const allInputs = document.querySelectorAll('.form-control');
  allInputs.forEach(input => {
    input.addEventListener('input', () => {
      input.classList.remove('error');
      const errorElement = input.parentElement.querySelector('.error-message');
      if (errorElement) {
        errorElement.remove();
      }
    });
  });

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