/**
 * Authentication Module
 * Handles user login, registration, and auth state
 */

import { state } from '../core/state.js';

/**
 * Show authentication modal
 */
export function showAuth() {
  document.getElementById('authModal').style.display = 'block';
}

/**
 * Close authentication modal
 */
export function closeAuth() {
  document.getElementById('authModal').style.display = 'none';
}

/**
 * Switch between login and register tabs
 */
export function switchAuthTab(tab) {
  if (tab === 'login') {
    document.getElementById('loginForm').style.display = 'block';
    document.getElementById('registerForm').style.display = 'none';
  } else {
    document.getElementById('loginForm').style.display = 'none';
    document.getElementById('registerForm').style.display = 'block';
  }
}

/**
 * Register new user
 */
export async function register() {
  const username = document.getElementById('regUsername').value;
  const email = document.getElementById('regEmail').value;
  const password = document.getElementById('regPassword').value;

  try {
    const response = await fetch("/auth/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, email, password })
    });

    const data = await response.json();

    if (data.success) {
      document.getElementById('authMessage').innerHTML = '<div class="success">Registration successful! Please login.</div>';
      setTimeout(() => switchAuthTab('login'), 2000);
    } else {
      document.getElementById('authMessage').innerHTML = `<div class="error">${data.error}</div>`;
    }
  } catch (error) {
    document.getElementById('authMessage').innerHTML = '<div class="error">Error registering</div>';
  }
}

/**
 * Login user
 */
export async function login() {
  const username = document.getElementById('loginUsername').value;
  const password = document.getElementById('loginPassword').value;

  try {
    const response = await fetch("/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password })
    });

    const data = await response.json();

    if (data.success) {
      state.authToken = data.token;
      state.currentUser = data.username;
      document.getElementById('username').textContent = state.currentUser;
      document.getElementById('userInfo').style.display = 'block';
      document.getElementById('authMessage').innerHTML = '<div class="success">Login successful!</div>';
      setTimeout(() => closeAuth(), 1500);
    } else {
      document.getElementById('authMessage').innerHTML = `<div class="error">${data.error}</div>`;
    }
  } catch (error) {
    document.getElementById('authMessage').innerHTML = '<div class="error">Error logging in</div>';
  }
}

// Make globally accessible for onclick handlers
window.showAuth = showAuth;
window.closeAuth = closeAuth;
window.switchAuthTab = switchAuthTab;
window.register = register;
window.login = login;
