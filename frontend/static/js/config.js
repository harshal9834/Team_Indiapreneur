/**
 * Verionyx AI - Frontend Configuration
 * 
 * If you are deploying the frontend to Vercel and the backend to Render:
 * 1. Update PROD_API_URL below with your Render backend URL.
 * 2. This config is loaded before other scripts and provides 
 *    the global API_BASE_URL.
 */

const CONFIG = {
    // DEV settings
    DEV_API_URL: 'http://127.0.0.1:5000',
    
    // PRODUCTION settings 
    // REPLACE this with your actual Render URL (e.g., 'https://verionyx-api.onrender.com')
    PROD_API_URL: 'https://verionyx-platform.onrender.com', 
    
    IS_PROD: window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1'
};

const API_BASE_URL = CONFIG.IS_PROD ? CONFIG.PROD_API_URL : CONFIG.DEV_API_URL;

console.log(`[Verionyx] Initialized in ${CONFIG.IS_PROD ? 'PRODUCTION' : 'DEVELOPMENT'} mode`);
console.log(`[Verionyx] API Endpoint: ${API_BASE_URL}`);
