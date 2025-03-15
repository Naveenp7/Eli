// Authentication Handling
const loginForm = document.getElementById('login-form');
const loginError = document.getElementById('login-error');
const registerForm = document.getElementById('register-form');
const registerError = document.getElementById('register-error');
const showRegisterLink = document.getElementById('show-register');
const showLoginLink = document.getElementById('show-login');
const loginSection = document.getElementById('login-section');
const registerSection = document.getElementById('register-section');

// Handle login form submission
if (loginForm) {
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
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password }),
            });

            const data = await response.json();

            if (response.ok) {
                localStorage.setItem('authToken', data.token);
                window.location.href = data.redirect || '/dashboard';
            } else {
                loginError.textContent = data.error || 'Login failed.';
            }
        } catch (error) {
            console.error('Error during login:', error);
            loginError.textContent = 'An error occurred. Please try again.';
        }
    });
}

// Handle registration form submission
if (registerForm) {
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
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password }),
            });

            const data = await response.json();

            if (response.ok) {
                registerError.textContent = 'Registration successful! Redirecting to login...';
                setTimeout(() => {
                    registerSection.classList.add('hidden');
                    loginSection.classList.remove('hidden');
                    registerError.textContent = '';
                }, 2000);
            } else {
                registerError.textContent = data.error || 'Registration failed.';
            }
        } catch (error) {
            console.error('Error during registration:', error);
            registerError.textContent = 'An error occurred. Please try again.';
        }
    });
}

// Toggle between login and register forms
if (showRegisterLink && loginSection && registerSection) {
    showRegisterLink.addEventListener('click', (e) => {
        e.preventDefault();
        loginSection.classList.add('hidden');
        registerSection.classList.remove('hidden');
    });
}

if (showLoginLink && loginSection && registerSection) {
    showLoginLink.addEventListener('click', (e) => {
        e.preventDefault();
        registerSection.classList.add('hidden');
        loginSection.classList.remove('hidden');
    });
}

// Check authentication status
function isAuthenticated() {
    return !!localStorage.getItem('authToken');
}

// Logout functionality
function logout() {
    fetch('/logout', { method: 'GET' })
        .then(() => {
            localStorage.removeItem('authToken');
            window.location.href = '/login';
        })
        .catch(error => {
            console.error('Logout error:', error);
            localStorage.removeItem('authToken');
            window.location.href = '/login';
        });
}

// Make functions available globally
window.isAuthenticated = isAuthenticated;
window.logout = logout;

// Initial auth check
document.addEventListener('DOMContentLoaded', () => {
    const currentPath = window.location.pathname;
    if (!isAuthenticated() && currentPath !== '/login' && currentPath !== '/register') {
        window.location.href = '/login';
    }
});