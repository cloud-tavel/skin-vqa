Page({
  data: {},

  takePhoto() {
    wx.chooseMedia({
      count: 1,
      mediaType: ['image'],
      sourceType: ['camera'],
      success: (res) => this.goAsk(res.tempFiles[0].tempFilePath)
    })
  },

  chooseFromAlbum() {
    wx.chooseMedia({
      count: 1,
      mediaType: ['image'],
      sourceType: ['album'],
      success: (res) => this.goAsk(res.tempFiles[0].tempFilePath)
    })
  },

  goAsk(path) {
    wx.navigateTo({ url: '/pages/ask/ask?img=' + encodeURIComponent(path) })
  },

  goHistory() {
    wx.navigateTo({ url: '/pages/history/history' })
  },

  goScience() {
    wx.navigateTo({ url: '/pages/knowledge/knowledge' })
  }
})