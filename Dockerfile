FROM centos:centos7 AS base-dependencies
LABEL maintainer "Abdelrahman Hosny <abdelrahman_hosny@brown.edu>"


# Install python dev
RUN yum install -y https://centos7.iuscommunity.org/ius-release.rpm && \
    yum update -y && \
    yum install -y python36u python36u-libs python36u-devel python36u-pip


RUN pip3 install --user numpy scipy matplotlib tensorflow pandas pytest

FROM base-dependencies AS builder

COPY . /OpeNPDN
WORKDIR /OpeNPDN
