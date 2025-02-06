export enum License {
  PUBLIC_DOMAIN = "Public Domain",
  CC_BY_3_0 = "CC BY 3.0",
  CC_BY_SA_4_0 = "CC BY-SA 4.0",
}


type AttributionProps = {
  author: string,
  license: License,
  via: string,
  href: string,
}

export function Attribution(props: AttributionProps) {
  return <small>
    {props.author}, via <a href={props.href}>{props.via}</a>, ({props.license})
  </small>
}
