import React from 'react'

export interface CitationProps {
  source: string,
  commentary?: string
}

export interface CitationBankProps {

}

export interface CitationComponents {
  Citation: React.FunctionComponent<CitationProps>,
  CitationBank: React.FunctionComponent<CitationBankProps>,
}

export interface CitationManagerProps {
  prefix: string,
  children: (c: CitationComponents) => React.ReactElement
}


class CitationManager extends React.Component<CitationManagerProps> {

  // maps indexes
  map = new Map<string, number>();

  constructor(props: CitationManagerProps) {
    super(props);
  }

  citation = (props: CitationProps) => {
    let index: number;
    if (this.map.has(props.source)) {
      index = this.map.get(props.source)!;
    } else {
      index = this.map.size + 1;
      this.map.set(props.source, index);
    }

    return <a href={'#' + this.props.prefix + index} style={{ textDecoration: "none" }} className="me-1">
      <sup>[{index}]</sup>
    </a>
  }

  citationBank = (props: CitationBankProps) => <ol>{
    Array.from(
      this.map,
      ([source, index]) =>
        <li key={index} id={this.props.prefix + index}>
          <a href={source}>{source}</a>
        </li>
    )
  }</ol>



  render() {
    return this.props.children({
      Citation: this.citation,
      CitationBank: this.citationBank
    })
  }

}

export default CitationManager;
