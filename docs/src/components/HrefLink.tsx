
type HrefLinkProps = {
  id?: string
  className?: string
  href: string
}

function HrefLink(props: HrefLinkProps) {
  return <a className={props.className} id={props.className} href={props.href}>{props.href}</a>
}

export default HrefLink;
