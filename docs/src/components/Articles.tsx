import ArticleLayout from '../components/ArticleLayout';
import Section from '../components/Section';

import { format } from 'date-fns';

import { Article, articleData } from '../components/ArticleData';

type ArticleLinkProps = {
  a: Article,
  showDate?: boolean
}

export const ArticleLink = ({ a, showDate }: ArticleLinkProps) => {
  if (showDate === true) {
    return <a href={a.url} children={`${a.name} (${format(a.published, "MMM d, y")})`} />
  } else {
    return <a href={a.url} children={a.name} />
  }
}

const Articles = () =>
  <Section id="articles" name="Articles">
    <ul>
      {Array.from(articleData.values()).map((a, i) =>
        <li key={i}><ArticleLink a={a} /></li>
      )}
    </ul>
  </Section>

export default Articles;
