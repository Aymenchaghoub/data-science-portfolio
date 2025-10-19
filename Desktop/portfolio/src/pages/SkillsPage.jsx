import React from 'react';
import './SkillsPage.css';

const SkillsPage = () => {
  const skills = [
    { name: 'Python', level: 'Advanced', category: 'Programming Languages', color: '#3776ab', years: '3+' },
    { name: 'C++', level: 'Intermediate', category: 'Programming Languages', color: '#00599c', years: '2+' },
    { name: 'PHP', level: 'Intermediate', category: 'Web Development', color: '#777bb4', years: '2+' },
    { name: 'JavaScript', level: 'Intermediate', category: 'Web Development', color: '#f7df1e', years: '2+' },
    { name: 'Node.js', level: 'Beginner', category: 'Web Development', color: '#339933', years: '1+' },
    { name: 'React.js', level: 'Beginner', category: 'Frontend', color: '#61dafb', years: '1+' },
    { name: 'Bootstrap', level: 'Intermediate', category: 'Frontend', color: '#7952b3', years: '2+' },
    { name: 'MySQL', level: 'Intermediate', category: 'Database', color: '#4479a1', years: '2+' },
    { name: 'Git', level: 'Intermediate', category: 'Tools', color: '#f05032', years: '2+' },
    { name: 'Data Science', level: 'Intermediate', category: 'Specialization', color: '#6a1b9a', years: '2+' },
    { name: 'Machine Learning', level: 'Beginner', category: 'Specialization', color: '#ff6b6b', years: '1+' },
    { name: 'Pandas', level: 'Intermediate', category: 'Data Analysis', color: '#150458', years: '2+' },
    { name: 'NumPy', level: 'Intermediate', category: 'Data Analysis', color: '#4dabcf', years: '2+' },
    { name: 'Scikit-learn', level: 'Beginner', category: 'Data Analysis', color: '#f7931e', years: '1+' }
  ];

  const categories = [...new Set(skills.map(skill => skill.category))];

  return (
    <div className="skills-page">
      <div className="container">
        <div className="page-header">
          <h1 className="page-title">Skills & Technologies</h1>
          <p className="page-subtitle">
            Here are the technologies and tools I work with, organized by category.
          </p>
        </div>

        <div className="skills-overview">
          <div className="overview-card">
            <h3>Total Skills</h3>
            <span className="overview-number">{skills.length}</span>
          </div>
          <div className="overview-card">
            <h3>Categories</h3>
            <span className="overview-number">{categories.length}</span>
          </div>
          <div className="overview-card">
            <h3>Years Learning</h3>
            <span className="overview-number">3+</span>
          </div>
        </div>

        <div className="skills-by-category">
          {categories.map((category, categoryIndex) => (
            <div key={category} className="skill-category fade-in" style={{ animationDelay: `${categoryIndex * 0.1}s` }}>
              <h2 className="category-title">{category}</h2>
              <div className="skills-grid">
                {skills
                  .filter(skill => skill.category === category)
                  .map((skill, skillIndex) => (
                    <div 
                      key={skill.name} 
                      className="skill-card"
                      style={{ 
                        animationDelay: `${(categoryIndex * 0.1) + (skillIndex * 0.05)}s`,
                        '--skill-color': skill.color 
                      }}
                    >
                      <div className="skill-header">
                        <h3 className="skill-name">{skill.name}</h3>
                        <div className="skill-meta">
                          <span className="skill-level">{skill.level}</span>
                          <span className="skill-years">{skill.years} years</span>
                        </div>
                      </div>
                      <div className="skill-progress">
                        <div 
                          className="progress-bar"
                          style={{ 
                            width: skill.level === 'Advanced' ? '90%' : 
                                   skill.level === 'Intermediate' ? '70%' : '50%'
                          }}
                        ></div>
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          ))}
        </div>

        <div className="learning-section">
          <h2>Currently Learning</h2>
          <div className="learning-items">
            <div className="learning-item">
              <span className="learning-icon">🚀</span>
              <span>React.js</span>
            </div>
            <div className="learning-item">
              <span className="learning-icon">☁️</span>
              <span>Cloud Computing</span>
            </div>
            <div className="learning-item">
              <span className="learning-icon">🤖</span>
              <span>Machine Learning</span>
            </div>
          </div>
        </div>

        <div className="skills-footer">
          <p>Want to discuss any of these technologies or collaborate on a project?</p>
          <a href="/contact" className="btn btn-primary">Let's Connect</a>
        </div>
      </div>
    </div>
  );
};

export default SkillsPage;
