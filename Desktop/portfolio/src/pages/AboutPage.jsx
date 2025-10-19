import React from 'react';
import './AboutPage.css';

const AboutPage = () => {
  return (
    <div className="about-page">
      <div className="container">
        <div className="page-header">
          <h1 className="page-title">About Me</h1>
          <p className="page-subtitle">
            Get to know more about my journey, interests, and aspirations.
          </p>
        </div>

        <div className="about-content">
          <div className="about-main fade-in">
            <div className="about-text">
              <h2>My Story</h2>
              <p>
                Hello! I'm Chaghoub Aymen, a passionate computer science and data science student 
                currently pursuing my studies at the University of Lille. My journey in technology 
                began with curiosity about how things work behind the scenes, and it has evolved 
                into a deep passion for creating innovative solutions.
              </p>
              <p>
                As a student, I've had the opportunity to work on various projects ranging from 
                web development to machine learning applications. Each project has been a learning 
                experience that has helped me grow both technically and personally. I believe in 
                the power of technology to solve real-world problems and make a positive impact 
                on society.
              </p>
              <p>
                My goal is to become a software engineer or data scientist, where I can apply my 
                technical skills to create meaningful solutions. I'm particularly interested in 
                the intersection of software development and data science, where I can leverage 
                both programming skills and analytical thinking to build intelligent systems.
              </p>
            </div>
          </div>

          <div className="about-details">
            <div className="detail-card fade-in">
              <div className="card-icon">🎓</div>
              <h3>Education</h3>
              <p>Computer Science & Data Science</p>
              <span>University of Lille</span>
            </div>

            <div className="detail-card fade-in">
              <div className="card-icon">🎯</div>
              <h3>Goals</h3>
              <p>Software Engineer or Data Scientist</p>
              <span>Creating innovative solutions</span>
            </div>

            <div className="detail-card fade-in">
              <div className="card-icon">💡</div>
              <h3>Interests</h3>
              <p>Web Development, Machine Learning</p>
              <span>Problem-solving & Innovation</span>
            </div>
          </div>

          <div className="timeline-section">
            <h2>My Journey</h2>
            <div className="timeline">
              <div className="timeline-item fade-in">
                <div className="timeline-marker"></div>
                <div className="timeline-content">
                  <h3>Started Programming</h3>
                  <p>Began learning Python and discovered my passion for coding</p>
                  <span className="timeline-date">2021</span>
                </div>
              </div>
              <div className="timeline-item fade-in">
                <div className="timeline-marker"></div>
                <div className="timeline-content">
                  <h3>University Studies</h3>
                  <p>Enrolled in Computer Science & Data Science program at University of Lille</p>
                  <span className="timeline-date">2022</span>
                </div>
              </div>
              <div className="timeline-item fade-in">
                <div className="timeline-marker"></div>
                <div className="timeline-content">
                  <h3>Project Development</h3>
                  <p>Created multiple projects including web apps and machine learning models</p>
                  <span className="timeline-date">2023</span>
                </div>
              </div>
              <div className="timeline-item fade-in">
                <div className="timeline-marker"></div>
                <div className="timeline-content">
                  <h3>Portfolio Creation</h3>
                  <p>Built this portfolio to showcase my skills and projects</p>
                  <span className="timeline-date">2024</span>
                </div>
              </div>
            </div>
          </div>

          <div className="values-section">
            <h2>My Values</h2>
            <div className="values-grid">
              <div className="value-item fade-in">
                <div className="value-icon">🔍</div>
                <h3>Curiosity</h3>
                <p>Always eager to learn new technologies and explore innovative solutions</p>
              </div>
              <div className="value-item fade-in">
                <div className="value-icon">🤝</div>
                <h3>Collaboration</h3>
                <p>Believe in the power of teamwork and knowledge sharing</p>
              </div>
              <div className="value-item fade-in">
                <div className="value-icon">⚡</div>
                <h3>Innovation</h3>
                <p>Strive to create solutions that make a positive impact</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AboutPage;
