import React, { useState, useEffect } from 'react';
import { ShieldCheck, ShieldAlert, X } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

export default function ConsentBanner() {
  const { consentTracking, updateConsent } = useAuth();
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const bannerSeen = localStorage.getItem('smartcart_consent_seen');
    if (!bannerSeen) {
      setVisible(true);
    }
  }, []);

  const handleAccept = () => {
    updateConsent(true);
    localStorage.setItem('smartcart_consent_seen', 'true');
    setVisible(false);
  };

  const handleDecline = () => {
    updateConsent(false);
    localStorage.setItem('smartcart_consent_seen', 'true');
    setVisible(false);
  };

  if (!visible) return null;

  return (
    <div className="fixed bottom-0 inset-x-0 z-40 bg-slate-900/95 text-white backdrop-blur border-t border-slate-800 p-4 shadow-2xl transition-all">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="flex items-start gap-3">
          <div className="p-2 bg-indigo-500/20 text-indigo-400 rounded-lg shrink-0 mt-0.5">
            <ShieldCheck className="w-5 h-5" />
          </div>
          <div>
            <h4 className="text-sm font-semibold text-white">SmartCart AI Personalization & Privacy</h4>
            <p className="text-xs text-slate-300 mt-1 max-w-3xl leading-relaxed">
              We use adaptive Multi-Armed Bandits (LinUCB) to learn your taste from browsing signals in real-time.
              If you decline, we strictly serve non-personalized popularity recommendations and store no personal behavioral events.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3 shrink-0">
          <button
            onClick={handleDecline}
            className="px-3.5 py-2 text-xs font-medium text-slate-300 hover:text-white bg-slate-800 hover:bg-slate-700 rounded-lg border border-slate-700 transition"
          >
            Decline (Popularity Only)
          </button>
          <button
            onClick={handleAccept}
            className="px-4 py-2 text-xs font-semibold text-white bg-indigo-600 hover:bg-indigo-500 rounded-lg shadow-sm transition"
          >
            Accept Adaptive Personalization
          </button>
          <button
            onClick={() => setVisible(false)}
            className="text-slate-400 hover:text-white ml-1 p-1"
            title="Dismiss"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
}

