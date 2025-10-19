import React from 'react';
import { Link } from 'react-router-dom';
import './HomePage.css';

const HomePage = () => {
  return (
    <div className="homepage">
      <div className="hero-section">
        <div className="hero-content">
          <div className="hero-text fade-in">
            <h1 className="hero-title">
              Hello, I'm <span className="highlight">Chaghoub Aymen</span>
            </h1>
            <p className="hero-subtitle">
              Student in Computer Science & Data Science
            </p>
            <p className="hero-description">
              Passionate about technology and innovation, I'm currently pursuing my studies 
              in Computer Science and Data Science. I love creating solutions that make a 
              difference and exploring the endless possibilities of programming.
            </p>
            <div className="hero-buttons">
              <Link to="/projects" className="btn btn-primary">
                View my projects
              </Link>
              <Link to="/contact" className="btn btn-secondary">
                Get in touch
              </Link>
            </div>
          </div>
          <div className="hero-image fade-in">
            <div className="profile-card">
              <div className="profile-avatar">
                <div className="avatar-placeholder">
                  <span>CA</span>
                </div>
              </div>
              <div className="profile-info">
                <h3>Chaghoub Aymen</h3>
                <p>Computer Science Student</p>
                <div className="profile-stats">
                  <div className="stat">
                    <span className="stat-number">4+</span>
                    <span className="stat-label">Projects</span>
                  </div>
                  <div className="stat">
                    <span className="stat-number">9+</span>
                    <span className="stat-label">Skills</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <div className="quick-links">
        <div className="container">
          <h2 className="section-title">Quick Navigation</h2>
          <div className="links-grid">
            <Link to="/projects" className="quick-link-card">
              <div className="card-icon">💼</div>
              <h3>Projects</h3>
              <p>Explore my latest work and projects</p>
            </Link>
            <Link to="/skills" className="quick-link-card">
              <div className="card-icon">🛠️</div>
              <h3>Skills</h3>
              <p>Technologies and tools I work with</p>
            </Link>
            <Link to="/about" className="quick-link-card">
              <div className="card-icon">👨‍💻</div>
              <h3>About</h3>
              <p>Learn more about my journey</p>
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HomePage;
