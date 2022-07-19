package dev.isteam.chatbot.utils

import java.net.URL
import java.net.URLClassLoader

class CompatibleClassLoader(urls: Array<URL>, parent: ClassLoader) : URLClassLoader(urls, parent) {
    fun addUrl(url: URL?) {
        super.addURL(url)
    }
}