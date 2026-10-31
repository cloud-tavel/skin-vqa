const app = getApp()
const recorder = wx.getRecorderManager()
const DEFAULT_QUESTION = '基于这张图片，请分析可能的诊断结果和建议。'

Page({
  data: {
    imgPath: '',
    question: '',
    loading: false,
    error: '',
    recording: false,
    transcribing: false,
    chats: []
  },

  onLoad(options) {
    if (options.img) {
      this.setData({ imgPath: decodeURIComponent(options.img) })
    }
    this.session = []
    this.initVoice()
  },

  onShow() {
    // 兜底：页面复用/异常时确保会话数组存在，避免 submit 里触 TypeError
    if (!this.session) this.session = []
  },

  initVoice() {
    recorder.onStart(() => this.setData({ recording: true }))

    recorder.onStop((res) => {
      this.setData({ recording: false })
      if (res.tempFilePath) {
        this.uploadVoice(res.tempFilePath)
      }
    })

    recorder.onError((err) => {
      this.setData({ recording: false })
      wx.showToast({ title: '录音失败：' + (err.errMsg || '请重试'), icon: 'none' })
    })
  },

  toggleRecord() {
    if (this.data.recording) {
      recorder.stop()
    } else {
      // mp3 + 16k 采样，匹配腾讯云一句话识别 16k_zh 引擎
      recorder.start({ duration: 30000, format: 'mp3', sampleRate: 16000 })
    }
  },

  uploadVoice(tempFilePath) {
    this.setData({ transcribing: true })
    wx.showLoading({ title: '识别中...' })
    wx.uploadFile({
      url: app.globalData.baseUrl + '/asr',
      filePath: tempFilePath,
      name: 'file',
      success: (res) => {
        wx.hideLoading()
        this.setData({ transcribing: false })
        try {
          const data = JSON.parse(res.data)
          if (data.text) {
            this.setData({ question: data.text })
            wx.showToast({ title: '已识别', icon: 'success' })
          } else {
            wx.showToast({ title: data.error || '未识别到内容', icon: 'none' })
          }
        } catch (e) {
          wx.showToast({ title: '识别失败，请重试', icon: 'none' })
        }
      },
      fail: () => {
        wx.hideLoading()
        this.setData({ transcribing: false })
        wx.showToast({ title: '上传失败，请确认后端已启动', icon: 'none' })
      }
    })
  },

  onInput(e) {
    this.setData({ question: e.detail.value })
  },

  repick() {
    wx.chooseMedia({
      count: 1,
      mediaType: ['image'],
      sourceType: ['camera', 'album'],
      success: (res) => {
        // 换图即换病例，清空当前会话
        this.session = []
        this.setData({ imgPath: res.tempFiles[0].tempFilePath, chats: [], error: '' })
      }
    })
  },

  submit() {
    if (this.data.loading) {
      return  // 防重入：上一次请求未返回时忽略新的提交，避免并发追加气泡错乱
    }
    if (!this.data.imgPath) {
      wx.showToast({ title: '请先选择图片', icon: 'none' })
      return
    }
    const question = this.data.question || DEFAULT_QUESTION
    const chats = this.data.chats.concat({ role: 'user', text: question })
    this.setData({ loading: true, error: '', chats, question: '' })
    wx.uploadFile({
      url: app.globalData.baseUrl + '/ask',
      filePath: this.data.imgPath,
      name: 'file',
      formData: {
        question,
        history: this.session.length ? JSON.stringify(this.session) : ''
      },
      success: (res) => {
        try {
          const data = JSON.parse(res.data)
          if (data.diagnosis) {
            this.session.push(
              { role: 'user', content: question },
              { role: 'assistant', content: data.answer }
            )
            this.setData({ chats: chats.concat({ role: 'assistant', text: data.answer, diagnosis: data.diagnosis, latency: data.latency_s, turns: data.turns }) })
            this.saveHistory(question, data)
          } else {
            this.setData({ error: '返回异常，请重试' })
          }
        } catch (e) {
          this.setData({ error: '返回格式异常，请重试' })
        }
        this.setData({ loading: false })
      },
      fail: () => {
        this.setData({ error: '请求失败，请确认后端服务已启动', loading: false })
      }
    })
  },

  saveHistory(question, data) {
    const list = wx.getStorageSync('ask_history') || []
    list.unshift({
      time: new Date().toLocaleString('zh-CN'),
      question,
      diagnosis: data.diagnosis,
      answer: data.answer,
      latency: data.latency_s
    })
    wx.setStorageSync('ask_history', list.slice(0, 50))
  }
})
