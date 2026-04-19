// API Configuration
const API_BASE_URL = window.location.origin;

// Auth state management for vanilla JS dashboard
const Auth = {
    isAuthenticated: false,
    user: null,

    init() {
        // Migration support: check for both old and new keys if we changed them
        const storedUser = localStorage.getItem('verionyx_user') || localStorage.getItem('user');
        const token = localStorage.getItem('verionyx_token') || localStorage.getItem('token');
        
        if (storedUser && token) {
            try {
                this.user = JSON.parse(storedUser);
                this.isAuthenticated = true;
                // Normalize keys
                localStorage.setItem('user', storedUser);
                localStorage.setItem('token', token);
            } catch (e) {
                console.error("Auth init error:", e);
                this.logout();
            }
        }
        this.updateUI();
    },

    login(userData, token) {
        localStorage.setItem('token', token);
        localStorage.setItem('user', JSON.stringify(userData));
        this.user = userData;
        this.isAuthenticated = true;
        this.updateUI();
    },

    logout() {
        localStorage.removeItem('token');
        localStorage.removeItem('user');
        localStorage.removeItem('verionyx_token');
        localStorage.removeItem('verionyx_user');
        this.user = null;
        this.isAuthenticated = false;
        
        // Notify backend of logout
        fetch(`${API_BASE_URL}/api/auth/logout`, { method: 'POST' }).finally(() => {
            this.updateUI();
            window.location.href = '/login.html';
        });
    },

    updateUI() {
        const profileContainer = document.getElementById('userProfile');
        if (!profileContainer) return;

        if (this.isAuthenticated && this.user) {
            const initial = this.user.name ? this.user.name.charAt(0).toUpperCase() : '?';
            
            // Update Avatar
            const avatar = profileContainer.querySelector('.user-avatar');
            if (avatar) avatar.innerText = initial;

            // Update Name
            const nameSpan = profileContainer.querySelector('.user-name');
            if (nameSpan) nameSpan.innerText = this.user.name;

            // Update Email in dropdown
            const emailSpan = document.getElementById('userEmail');
            if (emailSpan) emailSpan.innerText = this.user.email || 'No email set';

            // Ensure profile is visible, buttons hidden
            profileContainer.style.display = 'flex';
            let btnGroup = document.querySelector('.auth-btn-group');
            if (btnGroup) btnGroup.remove();
        } else {
            // Check if already has buttons
            if (document.querySelector('.auth-btn-group')) return;

            // Hide profile display
            profileContainer.style.display = 'none';
            
            const topNavActions = document.querySelector('.top-nav-actions');
            if (topNavActions) {
                const authGroup = document.createElement('div');
                authGroup.className = 'auth-btn-group';
                authGroup.innerHTML = `
                    <button class="auth-btn sign-in" onclick="window.location.href='/login.html'">Sign In</button>
                    <button class="auth-btn sign-up" onclick="window.location.href='/signup.html'">Sign Up</button>
                `;
                topNavActions.appendChild(authGroup);
            }
        }
    }
};

document.addEventListener('DOMContentLoaded', () => Auth.init());
