# coding: utf-8

import scrapy
from scrapy import FormRequest
from qalist.items import QalistItem


class QalistSpider(scrapy.Spider):
    name = "one"
    allowed_domain = ["istock.stocom.net"]

    start_urls = [
        "http://istock.stocom.net/wiki/doku.php?id=faq:start"
    ]

    def start_requests(self):
        cookies = {
            'DokuWiki': 'ST-1391-DNoFcVkGVvcAOFhxUuWj-istockticketauth'
        }
        return [FormRequest(self.start_urls[0], cookies=cookies, callback=self.parse)]

    def parse(self, response):
        url_head = "http://istock.stocom.net"
        cookies = {
            'DokuWiki': 'ST-1391-DNoFcVkGVvcAOFhxUuWj-istockticketauth'
        }
        for sel in response.xpath('//div[@class="level3"]/p/a[@class="wikilink1"]/@href').extract():
            url = "%s%s" % (url_head, sel)
            yield FormRequest(url, cookies=cookies, callback=self.parse_detail)

    @staticmethod
    def parse_detail(response):
        item = QalistItem()

        question = response.xpath('//div[@class="wrap_center wrap_round wrap_help plugin_wrap"]/p/text()').extract()
        answer = response.xpath('//div[@class="wrap_center wrap_round wrap_tip plugin_wrap"]/p/text()').extract()
        link = response.xpath('//div[@class="wrap_center wrap_round wrap_tip plugin_wrap"]/p/a[@class="urlextern"]/@href').extract()
        graph = response.xpath('//div[@class="wrap_center wrap_round wrap_tip plugin_wrap"]/p/a[@class="media"]/@href').extract()
        if link:
            answer.extend(link)
        question = ''.join(question)
        answer = ''.join(answer)
        item["question"] = question.replace("：".decode("utf-8"), "").replace("\t".decode("utf-8"), "").replace("\n", "").strip()
        item["answer"] = answer.replace("：".decode("utf-8"), "").replace("\t".decode("utf-8"), "").replace("\n", "").strip()

        yield item
