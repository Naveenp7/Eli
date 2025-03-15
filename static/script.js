// DOM Elements
const loginPage = document.getElementById('login-page');
const registerPage = document.getElementById('register-page');
const dashboardPage = document.getElementById('dashboard-page');
const loginForm = document.getElementById('login-form');
const registerForm = document.getElementById('register-form');
const loginError = document.getElementById('login-error');
const registerError = document.getElementById('register-error');
const registerLink = document.getElementById('register-link');
const loginLink = document.getElementById('login-link');

// Initialize page visibility
loginPage.classList.remove('hidden');
registerPage.classList.add('hidden');
dashboardPage.classList.add('hidden');

// Show Register Page
registerLink.addEventListener('click', (e) => {
  e.preventDefault();
  loginError.textContent = '';
  loginPage.classList.add('hidden');
  registerPage.classList.remove('hidden');
});

// Show Login Page
loginLink.addEventListener('click', (e) => {
  e.preventDefault();
  registerError.textContent = '';
  registerPage.classList.add('hidden');
  loginPage.classList.remove('hidden');
});

// Login Functionality
loginForm.addEventListener('submit', async (e) => {
  e.preventDefault();

  const username = document.getElementById('username').value;
  const password = document.getElementById('password').value;

  if (username.trim() === '' || password.length !== 4 || isNaN(Number(password))) {
    loginError.textContent = 'Invalid username or password. Password must be 4 digits.';
    return;
  }

  try {
    const response = await fetch('/login', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ username, password }),
    });

    const data = await response.json();

    if (response.ok) {
      // Successful login
      loginPage.classList.add('hidden');
      dashboardPage.classList.remove('hidden');
      console.log('User details:', data.user);
    } else {
      loginError.textContent = data.error;
    }
  } catch (error) {
    console.error('Error during login:', error);
    loginError.textContent = 'An error occurred. Please try again.';
  }
});

// Register Functionality
registerForm.addEventListener('submit', async (e) => {
  e.preventDefault();

  const username = document.getElementById('reg-username').value;
  const password = document.getElementById('reg-password').value;

  if (username.trim() === '' || password.length !== 4 || isNaN(Number(password))) {
    registerError.textContent = 'Invalid username or password. Password must be 4 digits.';
    return;
  }

  try {
    const response = await fetch('/register', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ username, password }),
    });

    const data = await response.json();

    if (response.ok) {
      // Successful registration
      registerError.textContent = '';
      alert('Registration successful! Please login.');
      registerPage.classList.add('hidden');
      loginPage.classList.remove('hidden');
    } else {
      registerError.textContent = data.error;
    }
  } catch (error) {
    console.error('Error during registration:', error);
    registerError.textContent = 'An error occurred. Please try again.';
  }
});

// Navigation Functionality
function navigate(feature) {
  const content = document.getElementById('content');
  switch (feature) {
    case 'chatbot':
      content.innerHTML = '<h2>Chatbot</h2><p>Ask me anything about electricity management!</p>';
      break;
    case 'calculator':
      content.innerHTML = '<iframe src="static/fullelctro.html" style="width:100%; height:100vh; border:none;"></iframe>';
      break;
    case 'contact':
      content.innerHTML = '<h2>Contact Us</h2><p>Reach out to us for support.</p>';
      break;
    case 'account':
      content.innerHTML = '<h2>Account Details</h2><p>View and manage your account details.</p>';
      break;
    default:
      content.innerHTML = '<h2>Welcome to the Dashboard</h2><p>Select a feature from the navigation above.</p>';
  }
}