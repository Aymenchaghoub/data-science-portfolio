import React from 'react';
import './ProjectsPage.css';
import pythonProjects from '../data/python-projects.json';

const ProjectsPage = () => {
  const projects = [
    {
      id: 1,
      title: "School Management System",
      description: "A comprehensive web application for managing school operations including student records, grades, attendance tracking, and administrative tasks. Features role-based access control and real-time notifications.",
      technologies: ["PHP", "MySQL", "HTML", "CSS", "JavaScript", "Bootstrap"],
      image: "🏫",
      features: ["Student Management", "Grade Tracking", "Admin Dashboard", "Database Integration", "Role-based Access"],
      githubUrl: "https://github.com/aymen-chaghoub/school-management-system",
      demoUrl: "#",
      status: "Completed"
    },
    {
      id: 2,
      title: "Netflix Movie Recommendation Engine",
      description: "Machine learning project that analyzes movie features and user preferences to provide personalized recommendations. Uses collaborative filtering and content-based filtering algorithms.",
      technologies: ["Python", "Pandas", "Scikit-learn", "NumPy", "Jupyter", "Matplotlib"],
      image: "🎬",
      features: ["Data Analysis", "Recommendation Models", "Feature Engineering", "Performance Metrics", "Data Visualization"],
      githubUrl: "https://github.com/aymen-chaghoub/netflix-recommendation-engine",
      demoUrl: "#",
      status: "Completed"
    },
    {
      id: 3,
      title: "Pacman Game with AI",
      description: "A classic Pacman game implementation with intelligent ghost AI, multiple levels, and modern graphics. Features pathfinding algorithms and dynamic difficulty adjustment.",
      technologies: ["Python", "Pygame", "AI Algorithms", "Game Development"],
      image: "👻",
      features: ["Game Physics", "AI Enemies", "Score System", "Level Progression", "Pathfinding"],
      githubUrl: "https://github.com/aymen-chaghoub/pacman-ai-game",
      demoUrl: "#",
      status: "Completed"
    },
    {
      id: 4,
      title: "E-commerce Platform",
      description: "A fully functional online shopping platform with product catalog, shopping cart, user authentication, and payment integration. Features responsive design and modern UI.",
      technologies: ["HTML", "CSS", "JavaScript", "Bootstrap", "Local Storage", "API Integration"],
      image: "🛒",
      features: ["Product Catalog", "Shopping Cart", "User Authentication", "Responsive Design", "Payment Gateway"],
      githubUrl: "https://github.com/aymen-chaghoub/ecommerce-platform",
      demoUrl: "#",
      status: "Completed"
    }
  ];

  return (
    <div className="projects-page">
      <div className="container">
        <div className="page-header">
          <h1 className="page-title">My Projects</h1>
          <p className="page-subtitle">
            Here are some of the projects I've worked on during my studies and personal development.
          </p>
        </div>

        <div className="projects-grid">
          {projects.map((project, index) => (
            <div key={project.id} className="project-card fade-in" style={{ animationDelay: `${index * 0.1}s` }}>
              <div className="project-image">
                <div className="project-icon">{project.image}</div>
              </div>
              <div className="project-content">
                <h3 className="project-title">{project.title}</h3>
                <p className="project-description">{project.description}</p>
                
                <div className="project-features">
                  <h4>Key Features:</h4>
                  <ul>
                    {project.features.map((feature, idx) => (
                      <li key={idx}>{feature}</li>
                    ))}
                  </ul>
                </div>

                <div className="project-technologies">
                  <h4>Technologies Used:</h4>
                  <div className="tech-tags">
                    {project.technologies.map((tech, idx) => (
                      <span key={idx} className="tech-tag">{tech}</span>
                    ))}
                  </div>
                </div>

                <div className="project-actions">
                  <div className="project-status">
                    <span className={`status-badge ${project.status.toLowerCase()}`}>
                      {project.status}
                    </span>
                  </div>
                  <div className="project-links">
                    <a 
                      href={project.githubUrl} 
                      target="_blank" 
                      rel="noopener noreferrer"
                      className="project-link github-link"
                      aria-label={`View ${project.title} on GitHub`}
                    >
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                      </svg>
                      View Code
                    </a>
                    {project.demoUrl !== "#" && (
                      <a 
                        href={project.demoUrl} 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="project-link demo-link"
                        aria-label={`View ${project.title} demo`}
                      >
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                          <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
                        </svg>
                        Live Demo
                      </a>
                    )}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="page-header" style={{ marginTop: '3rem' }}>
          <h2 className="page-title">Python Projects</h2>
          <p className="page-subtitle">Automatically discovered from the <code>projets py/</code> folder.</p>
        </div>

        <div className="projects-grid">
          {pythonProjects.map((project, index) => (
            <div key={project.id} className="project-card fade-in" style={{ animationDelay: `${index * 0.08}s` }}>
              <div className="project-image">
                <div className="project-icon">{project.image}</div>
              </div>
              <div className="project-content">
                <h3 className="project-title">{project.title}</h3>
                <p className="project-description">{project.description}</p>

                <div className="project-technologies">
                  <h4>Technologies Used:</h4>
                  <div className="tech-tags">
                    {project.technologies.map((tech, idx) => (
                      <span key={idx} className="tech-tag">{tech}</span>
                    ))}
                  </div>
                </div>

                <div className="project-actions">
                  <div className="project-status">
                    <span className={`status-badge ${project.status.toLowerCase()}`}>
                      {project.status}
                    </span>
                  </div>
                  <div className="project-links">
                    <a href={project.githubUrl} target="_blank" rel="noopener noreferrer" className="project-link github-link">
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                      </svg>
                      View Code
                    </a>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="projects-footer">
          <p>Interested in collaborating or have questions about any of these projects?</p>
          <a href="/contact" className="btn btn-primary">Get in Touch</a>
        </div>
      </div>
    </div>
  );
};

export default ProjectsPage;
