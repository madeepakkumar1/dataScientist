import requests
import smtplib
from http import HTTPStatus
from bs4 import BeautifulSoup as BS
import robots
URL = 'https://www.jpsc.gov.in'


def find_jpsc_jobs():
    jobs_list = []
    res = requests.get(URL)
    if res.status_code == HTTPStatus.OK:
        soup = BS(res.content, 'lxml')
        # latest_job = soup.find_all('ul', attrs={'id': 'ulid'})
        latest_job = soup.select('#ulid li')
        for jobs in latest_job:
            jobs_list.append([jobs.text.replace('\n', ''), URL + '/' + jobs.find('a').get('href')])

    jobs = '\n'.join([str(job) for job in jobs_list ])
    print(jobs)
    send_mail(jobs)


def send_mail(message):
    gmail_user = 'deepakjon31@gmail.com'
    gmail_password = 'Devi12345@'

    sent_from = gmail_user
    # to = ['siteshkumar536@gmail.com', 'pappu095@gmail.com']
    to = ['sahoosurabhi@gmail.com']
    subject = 'JPSC lates Jobs'
    body = 'Hi,\n\n Please find below JPSC jobs and do apply :) \n\n' + message + \
           '\n\n Thanks,\n Deepak\n\nThis is System generated mail, Have a good day and Enjoy!!!'

    email_text = """\
    From: %s
    To: %s
    Subject: %s
    
    %s
    """ % (sent_from, ", ".join(to), subject, body)

    try:
        server = smtplib.SMTP('smtp.gmail.com:587')
        server.starttls()
        server.login(gmail_user, gmail_password)
        server.sendmail(sent_from, to, email_text)
        server.quit()
        print('Email sent!')
    except Exception as e:
        print('Something went wrong...', e)


if __name__ == '__main__':
    # if robots.check('/ebooks/', 'http://www.google.com'):
    if robots.check(URL):
        find_jpsc_jobs()
       # print("Deepak")
    else:
        print("kumar")
    # find_jpsc_jobs()