import { Component, inject } from '@angular/core';
import { TranslateService } from '@ngx-translate/core';

@Component({
  selector: 'app-language-selector',
  templateUrl: './language-selector.component.html',
  styleUrl: './language-selector.component.css'
})
export class LanguageSelectorComponent {
  private translateService: TranslateService = inject(TranslateService);
  
  flagMap = {
    'de': 'assets/images/flags/flag_de.svg',
    'en': 'assets/images/flags/flag_en.svg'
  }

  dropdownOpen = false;

  selectedFlag = this.flagMap[this.translateService.currentLang];

  toggleDropdown() {
    this.dropdownOpen = !this.dropdownOpen;
  }

  selectFlag(lang: string) {
    this.selectedFlag = this.flagMap[lang];
    
    this.translateService.use(lang);
    localStorage.setItem('PlotExtract.language', lang);
    this.dropdownOpen = false;
  }
}
