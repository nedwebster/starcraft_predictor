import { useState, useEffect } from 'react'
import './App.css'
import sc2Logo from './assets/sc2.png'
import fallbackImage from './assets/sc2_plot.png'

function App() {
  const [image, setImage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [backendStatus, setBackendStatus] = useState<'checking' | 'healthy' | 'error'>('checking');
  const [showFallback, setShowFallback] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);
  const [modalImage, setModalImage] = useState<string | null>(null);

  useEffect(() => {
    // Check backend health on component mount
    checkBackendHealth();
  }, []);

  const checkBackendHealth = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/health');
      if (response.ok) {
        setBackendStatus('healthy');
      } else {
        setBackendStatus('error');
        setError('Backend server is not responding correctly');
      }
    } catch (err) {
      setBackendStatus('error');
      setError('Cannot connect to backend server. Make sure it is running on http://localhost:8000');
    }
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setLoading(true);
    setError(null);
    setShowFallback(false);

    const formData = new FormData();
    formData.append('file', file);

    try {
      console.log('Uploading file:', file.name);
      const response = await fetch('http://localhost:8000/api/score-replay', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Failed to score replay');
      }

      const data = await response.json();
      console.log('Received response from server');
      setImage(`data:image/png;base64,${data.image}`);
    } catch (err) {
      console.error('Error uploading file:', err);
      setError(err instanceof Error ? err.message : 'An error occurred while processing the replay');
      setShowFallback(true);
    } finally {
      setLoading(false);
    }
  };

  const handleImageClick = (img: string) => {
    setModalImage(img);
    setModalOpen(true);
  };

  const closeModal = () => {
    setModalOpen(false);
    setModalImage(null);
  };

  return (
    <div className="App">
      <div className="logo-container">
        <img src={sc2Logo} alt="StarCraft II Logo" className="sc2-logo" />
        <h1 className="predictor-title">Predictor</h1>
      </div>
      {backendStatus === 'checking' && <p>Checking backend connection...</p>}
      {backendStatus === 'error' && (
        <div className="error-message">
          <p>⚠️ {error}</p>
          <button onClick={checkBackendHealth}>Retry Connection</button>
        </div>
      )}
      <div className="upload-container">
        <input
          type="file"
          accept=".SC2Replay"
          onChange={handleFileUpload}
          disabled={loading || backendStatus !== 'healthy'}
        />
        {loading && <p>Processing replay...</p>}
        {error && <p className="error">{error}</p>}
        {(image || showFallback) && (
          <div className="result">
            <h2>Replay Analysis</h2>
            {image ? (
              <img
                src={image}
                alt="Replay analysis"
                style={{ cursor: 'zoom-in' }}
                onClick={() => handleImageClick(image)}
              />
            ) : (
              <div className="fallback-container">
                <img
                  src={fallbackImage}
                  alt="Example replay analysis"
                  style={{ cursor: 'zoom-in' }}
                  onClick={() => handleImageClick(fallbackImage)}
                />
                <p className="fallback-message">Showing example analysis</p>
              </div>
            )}
          </div>
        )}
      </div>
      {modalOpen && modalImage && (
        <div className="modal-overlay" onClick={closeModal}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <img src={modalImage} alt="Enlarged analysis" className="modal-image" />
            <button className="modal-close" onClick={closeModal}>&times;</button>
          </div>
        </div>
      )}
    </div>
  );
}

export default App
