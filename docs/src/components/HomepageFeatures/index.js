import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: 'Mixed Control',
    Svg: require('@site/static/img/AgentOrg-1.svg').default,
    description: (
      <>
        Enables agents to address diverse goals driven by user needs and builder objectives, fostering dynamic and goal-oriented interactions.
      </>
    ),
  },
  {
    title: 'Task Composition',
    Svg: require('@site/static/img/AgentOrg-2.svg').default,
    description: (
      <>
        Breaks down complex real-world tasks into modular, reusable components managed by individual workers, promoting efficiency and scalability.
      </>
    ),
  },
  {
    title: 'Human Intervention',
    Svg: require('@site/static/img/AgentOrg-3.svg').default,
    description: (
      <>
        Integrates human oversight and enables interactive refinement, ensuring critical decisions are accurate and user preferences are prioritized.
      </>
    ),
  },
  {
    title: 'Continual Learning',
    Svg: require('@site/static/img/AgentOrg-4.svg').default,
    description: (
      <>
        Allows agents to evolve and improve by learning from interactions, ensuring sustained relevance and effectiveness in dynamic environments.
      </>
    ),
  }
];

function Feature({Svg, title, description}) {
  return (
    <div className={clsx('col col--3')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
