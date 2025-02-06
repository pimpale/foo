import React from 'react';

type AsideCardProps = {
  title: string,
  id?: string,
  children: string | React.ReactElement | (string | React.ReactElement)[],
}


const AsideCard: React.FunctionComponent<AsideCardProps> = props =>
  <div className="card mx-auto mb-4" id={props.id} style={{maxWidth:"960px"}}>
    <div className="card-body">
      <h6 className="card-title text-decoration-underline">{props.title}</h6>
      <div className="card-text">{props.children}</div>
    </div>
  </div>

export default AsideCard;
