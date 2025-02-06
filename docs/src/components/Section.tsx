import React from 'react';

type SectionProps = {
  id: string
  name: string
  children: React.ReactNode
}

const Section: React.FunctionComponent<SectionProps> = props => {
  return <section className="mt-5" style={{ overflow: "hidden", position: "relative", }}>
    <span
      id={props.id}
      style={{
        position: "absolute",
        top: "-100px",
        visibility: "hidden",
      }}></span>
    <div>
      <a href={`#${props.id}`} className="float-end text-muted"><h3>#</h3></a>
      <h2>{props.name}</h2>
    </div>
    {props.children}
  </section>
}

export default Section;
