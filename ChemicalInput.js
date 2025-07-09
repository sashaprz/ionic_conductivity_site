import React, { useState } from 'react';

const ChemicalInput = ({ onPredict }) => {
  const [composition, setComposition] = useState('');
  const [isValid, setIsValid] = useState(true);

  const validateComposition = (value) => {
    const chemicalFormulaRegex = /^[A-Za-z0-9().\s]+$/;
    return chemicalFormulaRegex.test(value) && value.trim().length > 0;
  };

const handleSubmit = (e) => {
  e.preventDefault();
  console.log('Submitting:', composition); // <-- Add this
  onPredict(composition);
};

  const handleInputChange = (e) => {
    const value = e.target.value;
    setComposition(value);
    if (!isValid && value.trim().length > 0) {
      setIsValid(validateComposition(value));
    }
  };

  return (
    <div style={{ minHeight: '100vh', background: 'linear-gradient(to bottom right, #eff6ff, #c7d2fe)', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '1rem' }}>
      <div style={{ width: '100%', maxWidth: '600px', background: 'white', borderRadius: '1rem', padding: '2rem', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}>
        <h1 style={{ fontSize: '2rem', fontWeight: 'bold', color: '#1e293b', marginBottom: '1rem', textAlign: 'center' }}>
          Ionic Conductivity Predictor
        </h1>
        <p style={{ fontSize: '1.2rem', color: '#64748b', textAlign: 'center', marginBottom: '2rem' }}>
          Predict the ionic conductivity of solid state electrolytes using machine learning
        </p>
        <form onSubmit={handleSubmit}>
          <label htmlFor="composition" style={{ fontWeight: '500', display: 'block', marginBottom: '0.5rem' }}>
            Chemical Composition
          </label>
          <input
            id="composition"
            type="text"
            placeholder="e.g., Li7La3Zr2O12, Li10GeP2S12"
            value={composition}
            onChange={handleInputChange}
            style={{
              fontSize: '1rem',
              height: '2.5rem',
              width: '100%',
              border: isValid ? '1px solid #cbd5e1' : '1px solid #ef4444',
              borderRadius: '0.5rem',
              padding: '0.5rem',
              marginBottom: '0.5rem'
            }}
          />
          {!isValid && (
            <p style={{ color: '#ef4444', fontSize: '0.9rem' }}>
              Please enter a valid chemical formula
            </p>
          )}
          <p style={{ color: '#64748b', fontSize: '0.9rem', marginBottom: '1rem' }}>
            Enter standard chemical notation (e.g., subscripts as numbers)
          </p>
          <button
            type="submit"
            style={{
              width: '100%',
              height: '2.5rem',
              fontSize: '1rem',
              fontWeight: '600',
              background: '#2563eb',
              color: 'white',
              border: 'none',
              borderRadius: '0.5rem',
              cursor: 'pointer'
            }}
            disabled={!composition.trim()}
          >
            Predict Conductivity
          </button>
        </form>
        <div style={{ marginTop: '2rem', textAlign: 'center' }}>
          <p style={{ color: '#64748b', fontSize: '0.9rem' }}>
            Powered by extensive electrolyte databases including OBELiX and Liverpool Ionics
          </p>
        </div>
      </div>
    </div>
  );
};

export default ChemicalInput;
