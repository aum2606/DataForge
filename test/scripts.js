/**
 * DataForge - Synthetic Data Generation Platform
 * Main JavaScript File
 */

document.addEventListener('DOMContentLoaded', function() {
    // Header scroll effect
    const header = document.querySelector('header');
    
    function handleHeaderScroll() {
        if (window.scrollY > 50) {
            header.classList.add('scrolled');
        } else {
            header.classList.remove('scrolled');
        }
    }
    
    window.addEventListener('scroll', handleHeaderScroll);
    handleHeaderScroll(); // Initial check
    
    // Toggle mobile menu
    const menuToggle = document.querySelector('.menu-toggle');
    const nav = document.querySelector('nav');
    
    if (menuToggle) {
        menuToggle.addEventListener('click', () => {
            nav.classList.toggle('active');
        });
    }
    
    // Data type card selection
    const dataTypeCards = document.querySelectorAll('.data-type-card');
    
    dataTypeCards.forEach(card => {
        card.addEventListener('click', () => {
            dataTypeCards.forEach(c => c.classList.remove('selected'));
            card.classList.add('selected');
        });
    });
    
    // Control dropdowns
    const controlGroups = document.querySelectorAll('.control-group');
    
    controlGroups.forEach(group => {
        const label = group.querySelector('.control-label');
        if (label) {
            label.addEventListener('click', () => {
                // Close other dropdowns
                controlGroups.forEach(g => {
                    if (g !== group && g.classList.contains('active')) {
                        g.classList.remove('active');
                    }
                });
                
                // Toggle current dropdown
                group.classList.toggle('active');
            });
        }
    });
    
    // Volume slider
    const volumeSlider = document.getElementById('volume-slider');
    const volumeDisplay = document.querySelector('.volume-display');
    
    if (volumeSlider && volumeDisplay) {
        volumeSlider.addEventListener('input', () => {
            const value = parseInt(volumeSlider.value).toLocaleString();
            volumeDisplay.textContent = `${value} records`;
        });
    }
    
    // Add new schema field
    const addFieldButton = document.querySelector('.add-field');
    
    if (addFieldButton) {
        addFieldButton.addEventListener('click', () => {
            const schemaEditor = document.querySelector('.schema-editor');
            const newField = document.createElement('div');
            newField.className = 'schema-field';
            
            newField.innerHTML = `
                <input type="text" placeholder="Field name" class="field-name">
                <select class="field-type">
                    <option value="id">ID</option>
                    <option value="integer">Integer</option>
                    <option value="float">Float</option>
                    <option value="string" selected>String</option>
                    <option value="date">Date</option>
                    <option value="boolean">Boolean</option>
                </select>
                <button class="remove-field">×</button>
            `;
            
            schemaEditor.insertBefore(newField, addFieldButton);
            
            // Add event listener to the new remove button
            const removeButton = newField.querySelector('.remove-field');
            removeButton.addEventListener('click', () => {
                schemaEditor.removeChild(newField);
            });
        });
    }
    
    // Remove schema field
    const removeFieldButtons = document.querySelectorAll('.remove-field');
    
    removeFieldButtons.forEach(button => {
        button.addEventListener('click', () => {
            const field = button.parentElement;
            const schemaEditor = field.parentElement;
            schemaEditor.removeChild(field);
        });
    });
    
    // Dropdown item selection
    const dropdownItems = document.querySelectorAll('.dropdown-item');
    
    dropdownItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            
            // Find all items in the same dropdown
            const parent = item.parentElement.parentElement;
            const siblings = parent.querySelectorAll('.dropdown-item');
            
            // Remove active class from all siblings
            siblings.forEach(sibling => {
                sibling.classList.remove('active');
            });
            
            // Add active class to clicked item
            item.classList.add('active');
            
            // Close the dropdown
            const controlGroup = parent.parentElement;
            controlGroup.classList.remove('active');
        });
    });
    
    // Scroll reveal animation
    function handleScrollAnimation() {
        // Elements to animate on scroll
        const featuresSection = document.querySelector('.features');
        const featureCards = document.querySelectorAll('.feature-card');
        const ctaSection = document.querySelector('.cta');
        
        // Function to check if element is in viewport
        function isInViewport(element, offset = 150) {
            const rect = element.getBoundingClientRect();
            return (
                rect.top <= (window.innerHeight - offset) &&
                rect.bottom >= 0
            );
        }
        
        // Check visibility and add classes
        if (featuresSection && isInViewport(featuresSection)) {
            featuresSection.classList.add('visible');
            
            // Stagger animation for feature cards
            featureCards.forEach((card, index) => {
                setTimeout(() => {
                    card.classList.add('visible');
                }, 100 * index);
            });
        }
        
        if (ctaSection && isInViewport(ctaSection)) {
            ctaSection.classList.add('visible');
        }
    }
    
    // Initialize animations
    function initializeAnimations() {
        // Add scroll listener for revealing sections
        window.addEventListener('scroll', handleScrollAnimation);
        
        // Initial check for elements in viewport
        handleScrollAnimation();
        
        // For browsers that support Intersection Observer
        if ('IntersectionObserver' in window) {
            const appearOptions = {
                threshold: 0.15,
                rootMargin: "0px 0px -100px 0px"
            };
            
            const appearOnScroll = new IntersectionObserver(
                (entries, appearOnScroll) => {
                    entries.forEach(entry => {
                        if (!entry.isIntersecting) return;
                        entry.target.classList.add('visible');
                        appearOnScroll.unobserve(entry.target);
                    });
                }, 
                appearOptions
            );
            
            const animatableElements = document.querySelectorAll('.features, .feature-card, .cta');
            animatableElements.forEach(element => {
                appearOnScroll.observe(element);
            });
        }
    }
    
    // Initialize animations
    initializeAnimations();
    
    // Add scroll-down indicator and functionality
    const heroSection = document.querySelector('.hero');
    if (heroSection) {
        const scrollIndicator = document.createElement('div');
        scrollIndicator.className = 'scroll-indicator';
        scrollIndicator.innerHTML = '<div class="scroll-arrow"></div><div class="scroll-text">Scroll Down</div>';
        heroSection.appendChild(scrollIndicator);
        
        scrollIndicator.addEventListener('click', () => {
            const featuresSection = document.querySelector('.features');
            if (featuresSection) {
                featuresSection.scrollIntoView({ behavior: 'smooth' });
            }
        });
    }
}); {
            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        entry.target.classList.add('animate-in');
                        observer.unobserve(entry.target);
                    }
                });
            }, { threshold: 0.2 });
            
            animatableElements.forEach(element => {
                observer.observe(element);
            });
        } else {
            // Fallback for browsers without Intersection Observer
            animatableElements.forEach(element => {
                element.classList.add('animate-in');
            });
        }
    }
    
    // Initialize animations
    initializeAnimations();
});