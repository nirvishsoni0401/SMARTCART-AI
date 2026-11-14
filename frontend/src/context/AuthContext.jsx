import React, { createContext, useContext, useState, useEffect } from 'react';
import { api } from '../api/client';
import { useToast } from './ToastContext';

export const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [consentTracking, setConsentTracking] = useState(true);
  const { addToast } = useToast();

  useEffect(() => {
    const token = localStorage.getItem('smartcart_token');
    const savedConsent = localStorage.getItem('smartcart_consent');
    if (savedConsent !== null) {
      setConsentTracking(savedConsent === 'true');
    }

    if (token) {
      api
        .getMe()
        .then((res) => {
          setUser(res.data);
          setConsentTracking(res.data.consent_tracking);
        })
        .catch(() => {
          localStorage.removeItem('smartcart_token');
          setUser(null);
        })
        .finally(() => setLoading(false));
    } else {
      setLoading(false);
    }
  }, []);

  const login = async (email, password) => {
    try {
      const res = await api.login({ email, password });
      localStorage.setItem('smartcart_token', res.data.access_token);
      setUser(res.data.user);
      setConsentTracking(res.data.user.consent_tracking);
      addToast(`Welcome back, ${res.data.user.name}!`, 'success');
      return true;
    } catch (err) {
      const msg = err.response?.data?.detail || 'Login failed. Please check credentials.';
      addToast(msg, 'error');
      return false;
    }
  };

  const register = async (name, email, password, consent = true) => {
    try {
      const res = await api.register({
        name,
        email,
        password,
        consent_tracking: consent,
      });
      localStorage.setItem('smartcart_token', res.data.access_token);
      setUser(res.data.user);
      setConsentTracking(res.data.user.consent_tracking);
      addToast(`Account created! Welcome, ${res.data.user.name}!`, 'success');
      return true;
    } catch (err) {
      const msg = err.response?.data?.detail || 'Registration failed. Try again.';
      addToast(msg, 'error');
      return false;
    }
  };

  const logout = () => {
    localStorage.removeItem('smartcart_token');
    setUser(null);
    addToast('Logged out successfully.', 'info');
  };

  const updateConsent = async (consent) => {
    setConsentTracking(consent);
    localStorage.setItem('smartcart_consent', String(consent));
    if (user) {
      try {
        await api.updateConsent(consent);
      } catch (e) {
        console.error('Failed to update consent on backend:', e);
      }
    }
    addToast(
      consent
        ? 'Personalized AI recommendations enabled.'
        : 'Privacy mode active: popularity recommendations only.',
      'info'
    );
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        loading,
        login,
        register,
        logout,
        consentTracking,
        updateConsent,
        isAdmin: user?.role === 'admin',
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}

