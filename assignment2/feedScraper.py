import feedparser
import re

file = open("clickbait.txt", "w+")
include_descr = False

feeds = ["https://www.buzzfeed.com/omg.xml",
         "https://www.buzzfeed.com/lol.xml",
         "https://www.buzzfeed.com/fail.xml",
         "https://www.buzzfeed.com/win.xml",
         "https://www.buzzfeed.com/wtf.xml",
         "https://www.buzzfeed.com/index.xml"]

for i, feed_link in enumerate(feeds):
    f = 1
    words = 0
    while words < 100000:
        link = feed_link + "?page=" + str(f)
        print(link)
        print(words)
        feed = feedparser.parse(link)
        #feed_title = feed['feed']['title']
        feed_entries = feed.entries

        if len(feed.entries) < 1:
            break

        for entry in feed.entries:
            article_title = entry.title

            article_published_at = entry.published # Unicode string
            article_published_at_parsed = entry.published_parsed # Time object
            # print("Title: {}".format(article_title))
            # print("Published at {}".format(article_published_at) )
            # print("Description {}".format(article_description) )

            words += len(article_title.split())
            file.write(article_title + "\n")

            if include_descr:
                article_description = re.search(r'\<h1\>(.*)\<\/h1\>+', entry.description).group(1)
                words += len(article_description.split())
                file.write(article_description + "\n")

        f += 1

file.close()